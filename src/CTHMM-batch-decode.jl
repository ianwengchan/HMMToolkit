# VERSION 1: NO COVARIATES
function CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; ϵ = 0)

    # cur_all_subject_prob = 0.0

    ## precomputation
    time_interval_list = df.time_interval
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list)

    num_distinct_time = size(distinct_time_list, 1) # from all time series
    num_state = size(Q_mat, 1)

    # ## soft count table C(delta, k, l), denoted Etij, to be accumulated from Evij
    # Etij = zeros(num_distinct_time, num_state, num_state)

    group_df = groupby(df, :ID)
    num_time_series = size(group_df, 1)

    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)
    Etij_list = Array{Array{Float64, 3}}(undef, num_time_series)
    Svi_list = Array{Array{Float64}}(undef, num_time_series)

    @threads for g = 1:num_time_series

        seq_df = group_df[g]
        len_time_series = nrow(seq_df)
        seq_time_interval_list = collect(skipmissing(seq_df.time_interval))

        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (soft_decode == 1) # soft decoding
            data_emiss_prob_list = obs_seq_emiss_list[g][1]
            Svi, Evij, subject_log_prob = CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, π_list; ϵ)
            # for i in 1:num_state
            #     seq_df[:, string("Sv", i)] = Svi[:, i]
            # end
        else
            log_data_emiss_prob_list = obs_seq_emiss_list[g][2]
            state_seq, subject_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)
            # for i in 1:num_state
            #     seq_df[:, string("Sv", i)] = vec(map(x -> x == i ? 1 : 0, state_seq))
            # end
            Svi = zeros(len_time_series, num_state)
            for i in 1:num_state
                Svi[:, i] = vec(map(x -> x == i ? 1 : 0, state_seq))
            end
        end

        Svi_list[g] = Svi
        cur_all_subject_prob_list[g] = subject_log_prob  # log prob from either soft or hard decoding

        # ## accum all prob from all time series
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob # log prob from either soft or hard decoding
    
        ## soft count table C(delta, k, l), denoted Etij, to be accumulated from Evij
        Etij = zeros(num_distinct_time, num_state, num_state)

        ## group Evij to be Etij
        for v = 1:(len_time_series-1)

            ## find T in distinct_time_list and corresponding t_idx (only one)
            T = seq_time_interval_list[v]
            t_idx = findfirst(x -> x .== T, distinct_time_list)

            if (soft_decode == 1)
                Etij[t_idx, :, :] = Etij[t_idx, :, :] + Evij[v, :, :]
            else
                i = state_seq[v]
                j = state_seq[v+1]
                Etij[t_idx, i, j] = Etij[t_idx, i, j] + 1
            end
        end # v

        Etij_list[g] = Etij
        
    end # g

    cur_all_subject_prob = sum(cur_all_subject_prob_list)
    Etij_all = sum(Etij_list)

    Svi_full = reduce(vcat, Svi_list)
    # for i in 1:num_state
    #     df[:, string("Sv", i)] = Svi_temp[:, i]
    # end

    return Svi_full, cur_all_subject_prob, Etij_all
    
end


function CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; ϵ = 0)

    Svi_full, cur_all_subject_prob, Etij_all = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; ϵ)

    num_state = size(Q_mat, 1)
    for i in 1:num_state
        df[:, string("Sv", i)] = Svi_full[:, i]
    end

    return cur_all_subject_prob, Etij_all

end


function CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; ϵ = 0)

    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    cur_all_subject_prob = 0.0

    ## precomputation
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list)

    group_df = groupby(df, :ID)
    num_time_series = size(group_df, 1)

    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)

    @threads for g = 1:num_time_series

        seq_df = group_df[g]

        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (soft_decode == 1) # soft decoding
            data_emiss_prob_list = obs_seq_emiss_list[g][1]
            subject_log_prob = CTHMM_likelihood_forward(seq_df, data_emiss_prob_list, Q_mat, π_list; ϵ)
            # DO NOT UPDATE Svi
        else
            log_data_emiss_prob_list = obs_seq_emiss_list[g][2]
            state_seq, subject_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)
            # DO NOT UPDATE Svi
        end

        # ## accum all prob from all time series
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob # log prob from either soft or hard decoding

        cur_all_subject_prob_list[g] = subject_log_prob # log prob from either soft or hard decoding
        
    end # g

    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    return cur_all_subject_prob
    
end




# VERSION 2: WITH COVARIATES, using the no covariate functions
function CTHMM_batch_decode_Etij_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0)

    ## precomputation
    num_state = size(π_list, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_subject)
    Etij_list = Array{Array{Float64, 3}}(undef, num_subject)   # store Etij for each subject, i.e. a specific covariate combination
    # note that the length of t depends on the subject
    Svi_list = Array{Array{Float64}}(undef, num_subject)

    # cur_all_subject_prob = 0.0
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = CTHMM.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        Svi, subject_log_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df_n, response_list, Qn, π_list, state_list; ϵ)
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob
        Svi_list[n] = Svi
        cur_all_subject_prob_list[n] = subject_log_prob
        Etij_list[n] = Etij
    end

    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    Svi_full = reduce(vcat, Svi_list)

    return Svi_full, cur_all_subject_prob, Etij_list
    
end


function CTHMM_batch_decode_Etij_and_append_Svi_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0)

    Svi_full, cur_all_subject_prob, Etij_list = CTHMM_batch_decode_Etij_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ)

    num_state = size(π_list, 1)
    for i in 1:num_state
        df[:, string("Sv", i)] = Svi_full[:, i]
    end

    return cur_all_subject_prob, Etij_list

end


function CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0)

    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    num_state = size(π_list, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_subject)

    # cur_all_subject_prob = 0.0
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = CTHMM.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        subject_log_prob = CTHMM_batch_decode_for_subjects(soft_decode, df_n, response_list, Qn, π_list, state_list; ϵ)
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob
        cur_all_subject_prob_list[n] = subject_log_prob
    end

    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    return cur_all_subject_prob
    
end