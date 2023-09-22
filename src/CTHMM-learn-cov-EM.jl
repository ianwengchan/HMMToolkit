function CTHMM_learn_cov_EM(df, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, α_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = true)

    ## precomputation before iteration
    # only once in the whole EM:
    num_state = size(π_list_init, 1)
    num_dim = size(state_list_init, 1)

    group_df = groupby(df, :subject_ID)
    num_subject = size(group_df, 1)
    distinct_time_list = Array{Vector{Int64}}(undef, num_subject) # the distinct_time_list depends on the subject
    @threads for n = 1:num_subject
        distinct_time_list[n] = CTHMM_precompute_distinct_time_list(group_df[n].time_interval)
    end

    # for i = 1:num_state
    #     subject_df[:, string("tau", i)] = missings(Float64, num_subject)
    #     for j = 1:num_state
    #         subject_df[:, string("N", i, j)] = missings(Float64, num_subject)
    #     end
    # end

    
    # start EM iteration
    α = Base.copy(α_init)
    π_list = Base.copy(π_list_init)
    state_list = Base.copy(state_list_init)
    ll_em_old = -Inf
    ll_em = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
    iter = 0
    
    while (ll_em - ll_em_old > ϵ) && (iter < max_iter)
        
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df; need to compute Etij separately for each subject
        ll_em_temp, Etij_list = CTHMM_batch_decode_Etij_and_append_Svi_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating π_list: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em

        ## part 1b: learning α using distinct time grouping

        α_old = copy(α) .- Inf
        X = Matrix(subject_df[!, covariate_list])   # same for all elements of Q and across iterations

        α_iter = 0
        while (α_iter < α_max_iter) && (sum((α - α_old) .^ 2) > 1e-10)
            α_iter = α_iter + 1
            α_old = copy(α)

            # have to find Nij_mat and taui_list for each subject
            subject_df = CTHMM_learn_cov_nij_taui(num_state, num_subject, subject_df, covariate_list, distinct_time_list, α, Etij_list)

            k = 0
            for i in 1:(num_state-1)    # fill by row
                tau = subject_df[!, string("tau", i)]
                # w = coalesce.(tau, 0.0)
                for j in 1:num_state
                    if i != j
                        k = k + 1
                        y = log.(subject_df[!, string("N", i, j)] ./ tau)
                        α[k, :] = GLM.coef(GLM.lm(X, y))
                        # println(α)
                    end
                end
            end

            # k = 0
            # for i in 1:(num_state-1)    # fill by row
            #     tau = subject_df[!, string("tau", i)]
            #     w = coalesce.(tau, 0.0)
            #     for j in 1:num_state
            #         if i != j
            #             k = k + 1
            #             y = subject_df[!, string("N", i, j)] ./ tau
            #             α[k, :] = coef(glm(X, y, Gamma(), LogLink(), wts = w))
            #             println(α)
            #         end
            #     end
            # end
            
            ll_em = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(α_iter), updating α: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
                @info(
                    "New α: $(α)"
                )
            end
            ll_em_temp = ll_em
        end

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty=penalty)
                # Svi has not been changed since E-step
            end
            ll_em = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter), updating dim $(d): $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em
        end
        
    end # iter
    
    converge = (ll_em - ll_em_old > ϵ) ? false : true

    return (α_fit = α, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em)

end