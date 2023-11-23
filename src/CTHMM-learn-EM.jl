function CTHMM_learn_EM(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = true)

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

    GC.safepoint()
    
    # while (ll_em - ll_em_old > ϵ) && (iter < max_iter)
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
            
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)

        GC.safepoint()
        
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

        Q_old = copy(Q_mat) .- Inf

        Q_iter = 0
        while (Q_iter < Q_max_iter) # && (sum((Q_mat - Q_old) .^ 2) > 1e-10)
            Q_iter = Q_iter + 1
            Q_old = copy(Q_mat)
            distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
            Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
            Q_mat = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

            GC.safepoint()

            ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(Q_iter), updating Q: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em
        end

        GC.safepoint()

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            # params_old = CTHMM.params.(state_list[d, :])
            params_old = (x -> round.(x, digits = 4)).(CTHMM.params.(state_list[d, :]))
            for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty=penalty)
                # Svi has not been changed since E-step
            end

            ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter), updating dim $(d): $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
                if s == "-"
                    params_new = (x -> round.(x, digits = 4)).(CTHMM.params.(state_list[d, :]))
                    @info(
                        "Intended update of params: $(params_old) ->  $(params_new)"
                    )
                end
            end
            ll_em_temp = ll_em
        end

        GC.safepoint()
        
    end # iter
    
    converge = (0 <= ll_em - ll_em_old <= ϵ)

    # converge = (ll_em - ll_em_old > ϵ) ? false : true

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em)

end


function CTHMM_learn_EM_Q_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1)

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
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        
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

        Q_old = copy(Q_mat) .- Inf

        Q_iter = 0
        while (Q_iter < Q_max_iter) # && (sum((Q_mat - Q_old) .^ 2) > 1e-10)
            Q_iter = Q_iter + 1
            Q_old = copy(Q_mat)
            distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
            Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
            Q_mat = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

            ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(Q_iter), updating Q: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em
        end
        
    end # iter
    
    converge = (ll_em - ll_em_old > ϵ) ? false : true

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em)

end


function CTHMM_learn_EM_expert_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1)

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
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
        
        # ## M-step, with last estimated parameters
        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty=true)
                # Svi has not been changed since E-step
            end
            ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
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

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em)

end