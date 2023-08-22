function CTHMM_learn_EM(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, soft_decode = 1)

    ## precomputation before iteration
    # only once in the whole EM:
    distinct_time_list = CTHMM_precompute_distinct_time_list(df.time_interval)
    num_state = size(Q_mat_init, 1)
    num_dim = size(state_list_init, 1)
    num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series
    
    # start EM iteration
    Q_mat = Base.copy(Q_mat_init)
    π_list = Base.copy(π_list_init)
    state_list = Base.copy(state_list_init)
    ll_em_old = -Inf
    ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
    iter = 0
    
    while (ll_em - ll_em_old > ϵ) && (iter < max_iter)
        
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating π_list: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em

        ## part 1b: learning Q_mat using distinct time grouping
        distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
        Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
        Q_mat = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

        ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating Q: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)])
                # Svi has not been changed since E-step
            end
        end

        ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating distributions: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em        
        
        # ## check if reached a fixed point
        # [is_termindate] = CTHMM_learn_decide_termination[cur_all_subject_prob, pre_all_subject_prob];    
        # if (is_termindate .== 1)
        #     str = sprintf("#s/num_iter.txt", out_dir)
        #     fp = fopen(str, "wt")
        #     fprintf(fp, "#d\n", model_iter_count)
        #     fclose(fp)
        #     if (cur_all_subject_prob .< pre_all_subject_prob)
        #         Q_mat = pre_Q_mat
        #     end
        #     break()
        # end
        
        # ## store current all subject prob
        # pre_all_subject_prob = cur_all_subject_prob
        
    end # iter
    
    converge = (ll_em - ll_em_old > ϵ) ? false : true

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em)

end