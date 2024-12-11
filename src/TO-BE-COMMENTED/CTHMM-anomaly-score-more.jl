function CTHMM_batch_anomaly_score_for_cov_next_obs(uniform, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0, keep_last = 1, ZI_list = nothing)
    
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)
    
    # precomputation
    num_state = size(π_list, 1)
    
    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    forecast_list_cov = Array{Array{Array{Float64}}}(undef, num_subject)
    
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        forecast_list_cov[n] = CTHMM_batch_anomaly_score_for_next_obs(uniform, df_n, response_list, Qn, π_list, state_list; ϵ, keep_last, ZI_list)
        GC.safepoint()
    end    
            
    GC.safepoint()

    return forecast_list_cov
    
end


function CTHMM_anomaly_test_and_percentage_uni(AD, forecast_list; threshold = 0.05)
    
    # simplified function for univariate anomaly detection
    
    # AD = 1 if Anderson-Darling test (stronger); AD = 0 if Jarque-Bera test
    # default threshold (significance level) is 0.05
    
    num_trips = length(forecast_list)
    num_dim = size(forecast_list[1], 2)
    # pvalue_list_separate = Array{Array{Float64}}(undef, num_trips)
    pvalue_list_combined = Array{Float64}(undef, num_trips)
    dof = 2*num_dim
    
    if (AD == 1) # Anderson-Darling test
        
        @threads for g = 1:num_trips
            
            pvalue_list_combined[g] = pvalue(HypothesisTests.OneSampleADTest(vec(forecast_list[g]), Normal(0,1)))
            
        end
        
    else # Jarque-Bera test
        
        @threads for g = 1:num_trips
            
            pvalue_list_combined[g] = pvalue(HypothesisTests.JarqueBeraTest(vec(forecast_list[g])))
            
        end

    end
    
    anomaly_percentage = sum(pvalue_list_combined .<= threshold) / num_trips
    GC.safepoint()
    
    return pvalue_list_combined, anomaly_percentage
    
end




function CTHMM_anomaly_test_and_percentage(AD, com_method, forecast_list; threshold = 0.05)
    
    # AD = 1 if Anderson-Darling test (stronger); AD = 0 if Jarque-Bera test
    # com_method = 1 if Fisher's method; = 0 if harmonic mean
    # default threshold (significance level) is 0.05
    
    num_trips = length(forecast_list)
    num_dim = size(forecast_list[1], 2)
    pvalue_list_separate = Array{Array{Float64}}(undef, num_trips)
    pvalue_list_combined = Array{Float64}(undef, num_trips)
    dof = 2*num_dim
    
    if (AD == 1) # Anderson-Darling test
        
        @threads for g = 1:num_trips
            tmp = fill(0.0, num_dim)
            
            for d = 1:num_dim
                tmp[d] = pvalue(HypothesisTests.OneSampleADTest(vec(forecast_list[g][:, d]), Normal(0,1)))
            end
            
            pvalue_list_separate[g] = tmp
            
            if (com_method == 1)
                pvalue_list_combined[g] = Distributions.ccdf(Distributions.Chisq(dof), -2*sum(log.(tmp)))
            else
                pvalue_list_combined[g] = num_dim / sum(1 ./ tmp)
            end
        end
        
    else # Jarque-Bera test

        @threads for g = 1:num_trips
            tmp = fill(0.0, num_dim)

            for d = 1:num_dim
                tmp[d] = pvalue(HypothesisTests.JarqueBeraTest(vec(forecast_list[g][:, d])))
            end
            
            pvalue_list_separate[g] = tmp
            
            if (com_method == 1)
                pvalue_list_combined[g] = Distributions.ccdf(Distributions.Chisq(dof), -2*sum(log.(tmp)))
            else
                pvalue_list_combined[g] = num_dim / sum(1 ./ tmp)
            end
        end

    end
    
    anomaly_percentage = sum(pvalue_list_combined .<= threshold) / num_trips  # combined p-values
    GC.safepoint()
    
    return pvalue_list_separate, pvalue_list_combined, anomaly_percentage
    
end




function CTHMM_batch_anomaly_test_and_percentage(AD, forecast_list_cov; threshold = 0.05)
    
    # AD = 1 if Anderson-Darling test (stronger); AD = 0 if Jarque-Bera test
    # default threshold (significance level) is 0.05
    num_subject = length(forecast_list_cov)
    pvalue_list_cov_separate = Array{Array{Array{Float64}}}(undef, num_subject)
    pvalue_list_cov_combined = Array{Array{Float64}}(undef, num_subject)
    anomaly_percentage_list_cov = Array{Float64}(undef, num_subject)
    
    @threads for n = 1:num_subject
        
        forecast_list = forecast_list_cov[n]
        pvalue_list_cov_separate[n], pvalue_list_cov_combined[n], anomaly_percentage_list_cov[n] = CTHMM_anomaly_test_and_percentage(AD, forecast_list; threshold)
        
    end
    
    return pvalue_list_cov_separate, pvalue_list_cov_combined, anomaly_percentage_list_cov
    
end




function CTHMM_anomaly_score_for_next_obs_rolling(windowsize, uniform, seq_df, data_emiss_prob_list, data_emiss_cdf_list, Q_mat, π_list; ϵ = 0)
    
    # compute the posterior probability conditioned on last few observations, according to windowsize
    # adjusting the anomaly score function
    
    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # already feed-in the time intervals
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
    
    num_window = maximum([len_time_series - (windowsize + 1), 0])  # only need to adjust if there is at least 1 RW
    if (num_window > 0)
        time_since_list = collect(skipmissing(seq_df.time_since[2:(1+num_window)]))
        distinct_time_since_list = CTHMM_precompute_distinct_time_list(time_since_list)
        distinct_time_since_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_since_list, Q_mat)
        
        ## precomputing of Pt for every timepoint
        Pt_since_list = Array{Matrix{Float64}}(undef, num_window)
        @threads for v = 1:(num_window)
            T = time_since_list[v]
            t_idx = findfirst(x -> x .== T, distinct_time_since_list)
            Pt_since_list[v] = distinct_time_since_Pt_list[t_idx]
            GC.safepoint()
        end
        GC.safepoint()
    end

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    C = zeros(len_time_series, 1) # rescaling factor
    forecast = zeros((len_time_series - 1), 1) # store the forecast pseudo-residuals

    GC.safepoint()

    ## precomputing of Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end

    GC.safepoint()

    ## init alpha for time 1
    ALPHA[1, :] = transpose(π_list) * Diagonal(data_emiss_prob_list[1, :])
    C[1] = 1.0 / (sum(ALPHA[1, :]) + ϵ)
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    if (num_window <= 0)  # no rolling window; usual anomaly score computation
    
        for v = 2:len_time_series
        
            forecast[v-1] = maximum([0, minimum([1, sum(transpose(ALPHA[v-1, : ]) * Pt_list[v-1] * Diagonal(data_emiss_cdf_list[v, :]))])]) # divide by sum(ALPHA[v,:]) = 1
            ALPHA[v, :] = transpose(ALPHA[v-1, : ]) * Pt_list[v-1] * Diagonal(data_emiss_prob_list[v, :])

            # scaling
            C[v] = 1.0 / (sum(ALPHA[v, :]) + ϵ)
            ALPHA[v, :] = ALPHA[v, :] .* C[v]
        end # v
        GC.safepoint()
        
    else
        
        for v = 2:len_time_series
            
            if (v <= (windowsize + 1))  # before rolling window; usual computation
                
                forecast[v-1] = maximum([0, minimum([1, sum(transpose(ALPHA[v-1, : ]) * Pt_list[v-1] * Diagonal(data_emiss_cdf_list[v, :]))])]) # divide by sum(ALPHA[v,:]) = 1
                ALPHA[v, :] = transpose(ALPHA[v-1, : ]) * Pt_list[v-1] * Diagonal(data_emiss_prob_list[v, :])

                # scaling
                C[v] = 1.0 / (sum(ALPHA[v, :]) + ϵ)
                ALPHA[v, :] = ALPHA[v, :] .* C[v]
            
            else  # rolling window, adjustment at the window start
                
                u = v - (windowsize + 1)  # which rolling window
                # w = v - windowsize        # start of rolling window
                tmp = zeros((windowsize + 1), num_state)  # adjusted ALPHA
                tmp[1, :] = transpose(π_list) * Pt_since_list[u] * Diagonal(data_emiss_prob_list[(u+1), :])
                
                for idx = 2:windowsize
                    tmp[idx, :] = transpose(tmp[idx-1, : ]) * Pt_list[(u-1+idx)] * Diagonal(data_emiss_prob_list[(u+idx), :])
                end
                
                idx = windowsize + 1
                tmp[idx, :] = transpose(tmp[idx-1, : ]) * Pt_list[(u-1+idx)] * Diagonal(data_emiss_cdf_list[(u+idx), :])
                
                forecast[v-1] = maximum([0, minimum([1, sum(tmp[windowsize + 1, :])/sum(tmp[windowsize, :])])])                
            end
            
        end
        
    end
    
    if (uniform == 0) # normal
        forecast = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), forecast)
    end

    return forecast
    
end