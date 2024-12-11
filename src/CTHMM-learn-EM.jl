"""
    CTHMM_learn_EM(df, response_list, Q_mat_init, π_list_init, state_list_init;
        ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

Fit a CTHMM model via EM algorithm.

# Arguments
- `df`: Dataframe with (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat_init`: Initial estimate of the transition rate matrix `Q_mat`.
- `π_list_init`: Initial estimate of the initial state probabilities `π_list`.
- `state_list_init`: Initial estimate of the state dependent distributions `state_list`.

# Optional Arguments
- `ϵ`: Stopping criterion on loglikelihood, stop when the increment is less than ϵ. Default to 0.001.
- `max_iter`: Maximum number of iterations of the EM algorithm. Default to 200.
- `Q_max_iter`: Maximum number of iterations when updating `Q_mat`. Default to 5.
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `print_steps`: Logs parameter updates every (print_steps) iterations of the EM algorithm. Default to 1.
- `penalty`: `true` or `false`(default), indicating whether penalty is imposed on the magnitude of state dependent distribution parameters.
- `pen_params`: an array of penalty term on the magnitude of state dependent distribution parameters.
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `group_by_col`: List of columns to group the dataframe. If nothing is provided, group by "ID" by default.

# Return Values
- `Q_mat_fit`: Fitted transition rate matrix `Q_mat`.
- `π_list_fit`: Fitted initial state probabilities `π_list`.
- `state_list_fit`: Fitted state dependent distributions `state_list`.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting procedure.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np``: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""

function CTHMM_learn_EM(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

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

    # initialize pen_params if not provided OR penalty is false
    if penalty == false
        pen_params = [HMMToolkit.no_penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    elseif isnothing(pen_params)
        pen_params = [HMMToolkit.penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    end

    ll_em_old = -Inf
    ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
    ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
    ll_em = ll_em_np + ll_em_penalty
    iter = 0

    GC.safepoint()
    
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
            
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (soft_decode = 1), saving Svi to df
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)

        GC.safepoint()
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
        ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
        ll_em = ll_em_np + ll_em_penalty

        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating π_list: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em

        GC.safepoint()

        ## part 1b: learning Q_mat using distinct time grouping

        Q_old = copy(Q_mat) .- Inf

        Q_iter = 0
        while (Q_iter < Q_max_iter) # && (sum((Q_mat - Q_old) .^ 2) > 1e-10)
            Q_iter = Q_iter + 1
            Q_old = Base.copy(Q_mat)
            distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
            Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
            Q_mat = CTHMM_learn_update_Q_mat(Nij_mat, taui_list; block_size = block_size)

            GC.safepoint()

            ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
            ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(Q_iter), updating Q: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em
            GC.safepoint()
        end

        GC.safepoint()

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            params_old = (x -> round.(x, digits = 4)).(HMMToolkit.params.(state_list[d, :]))
            for i in 1:num_state
                state_list[d, i] = HMMToolkit.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty = penalty, pen_params_jk = pen_params[d][i])
                # Svi has not changed since E-step
            end

            ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
            ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter), updating dim $(d): $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
                if s == "-"
                    params_new = (x -> round.(x, digits = 4)).(HMMToolkit.params.(state_list[d, :]))
                    @info(
                        "Intended update of params: $(params_old) ->  $(params_new)"
                    )
                end
            end
            ll_em_temp = ll_em
            GC.safepoint()
        end

        GC.safepoint()
        
    end # iter
    
    converge = (0 <= ll_em - ll_em_old <= ϵ)

    num_params = _count_π(π_list) + _count_Q_mat(Q_mat) + _count_params(state_list)
    AIC = -2.0 * ll_em_np + 2 * num_params
    BIC = -2.0 * ll_em_np + log(nrow(df)) * num_params

    GC.safepoint()

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, 
            ll = ll_em, ll_np = ll_em_np,
            AIC = AIC, BIC = BIC)

end



"""
    CTHMM_learn_EM_Q_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
        ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

Update only the initial state probabilities and transition rate matrix of a CTHMM model via EM algorithm (sub-optimal to full model fitting).

# Arguments
- `df`: Dataframe with (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat_init`: Initial estimate of the transition rate matrix `Q_mat`.
- `π_list_init`: Initial estimate of the initial state probabilities `π_list`.
- `state_list_init`: Initial estimate of the state dependent distributions `state_list`.

# Optional Arguments
- `ϵ`: Stopping criterion on loglikelihood, stop when the increment is less than ϵ. Default to 0.001.
- `max_iter`: Maximum number of iterations of the EM algorithm. Default to 200.
- `Q_max_iter`: Maximum number of iterations when updating `Q_mat`. Default to 5.
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `print_steps`: Logs parameter updates every (print_steps) iterations of the EM algorithm. Default to 1.
- `penalty`: `true` or `false`(default), indicating whether penalty is imposed on the magnitude of state dependent distribution parameters.
- `pen_params`: an array of penalty term on the magnitude of state dependent distribution parameters.
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `group_by_col`: List of columns to group the dataframe. If nothing is provided, group by "ID" by default.

# Return Values
- `Q_mat_fit`: Fitted transition rate matrix `Q_mat`.
- `π_list_fit`: Fitted initial state probabilities `π_list`.
- `state_list_fit`: Fitted state dependent distributions `state_list`; should be same as `state_list_init.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting procedure.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np``: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""

function CTHMM_learn_EM_Q_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

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

    # initialize pen_params if not provided OR penalty is false
    if penalty == false
        pen_params = [HMMToolkit.no_penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    elseif isnothing(pen_params)
        pen_params = [HMMToolkit.penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    end

    ll_em_old = -Inf
    ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
    ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
    ll_em = ll_em_np + ll_em_penalty
    iter = 0

    GC.safepoint()
    
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
            
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (soft_decode = 1), saving Svi to df
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)

        GC.safepoint()
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
        ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
        ll_em = ll_em_np + ll_em_penalty

        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating π_list: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em

        GC.safepoint()

        ## part 1b: learning Q_mat using distinct time grouping

        Q_old = copy(Q_mat) .- Inf

        Q_iter = 0
        while (Q_iter < Q_max_iter) # && (sum((Q_mat - Q_old) .^ 2) > 1e-10)
            Q_iter = Q_iter + 1
            Q_old = Base.copy(Q_mat)
            distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
            Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
            Q_mat = CTHMM_learn_update_Q_mat(Nij_mat, taui_list; block_size = block_size)

            GC.safepoint()

            ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
            ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(Q_iter), updating Q: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em
            GC.safepoint()
        end

        GC.safepoint()
        
    end # iter
    
    converge = (0 <= ll_em - ll_em_old <= ϵ)

    num_params = _count_π(π_list) + _count_Q_mat(Q_mat) + _count_params(state_list)
    AIC = -2.0 * ll_em_np + 2 * num_params
    BIC = -2.0 * ll_em_np + log(nrow(df)) * num_params

    GC.safepoint()

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, 
            ll = ll_em, ll_np = ll_em_np,
            AIC = AIC, BIC = BIC)

end



"""
    CTHMM_learn_EM_state_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
        ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

Update only the state dependent distribution parameters of a CTHMM model via EM algorithm (sub-optimal to full model fitting).

# Arguments
- `df`: Dataframe with (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat_init`: Initial estimate of the transition rate matrix `Q_mat`.
- `π_list_init`: Initial estimate of the initial state probabilities `π_list`.
- `state_list_init`: Initial estimate of the state dependent distributions `state_list`.

# Optional Arguments
- `ϵ`: Stopping criterion on loglikelihood, stop when the increment is less than ϵ. Default to 0.001.
- `max_iter`: Maximum number of iterations of the EM algorithm. Default to 200.
- `Q_max_iter`: Maximum number of iterations when updating `Q_mat`. Default to 5.
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `print_steps`: Logs parameter updates every (print_steps) iterations of the EM algorithm. Default to 1.
- `penalty`: `true` or `false`(default), indicating whether penalty is imposed on the magnitude of state dependent distribution parameters.
- `pen_params`: an array of penalty term on the magnitude of state dependent distribution parameters.
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `group_by_col`: List of columns to group the dataframe. If nothing is provided, group by "ID" by default.

# Return Values
- `Q_mat_fit`: Fitted transition rate matrix `Q_mat`; should be same as `Q_mat_init`.
- `π_list_fit`: Fitted initial state probabilities `π_list`; should be same as `π_list_init`.
- `state_list_fit`: Fitted state dependent distributions `state_list`.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting procedure.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np``: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""

function CTHMM_learn_EM_state_only(df, response_list, Q_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, Q_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

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

    # initialize pen_params if not provided OR penalty is false
    if penalty == false
        pen_params = [HMMToolkit.no_penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    elseif isnothing(pen_params)
        pen_params = [HMMToolkit.penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    end

    ll_em_old = -Inf
    ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
    ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
    ll_em = ll_em_np + ll_em_penalty
    iter = 0

    GC.safepoint()
    
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
            
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (soft_decode = 1), saving Svi to df
        ll_em_temp, Etij = CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)

        GC.safepoint()
        
        ## M-step, with last estimated parameters
        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            params_old = (x -> round.(x, digits = 4)).(HMMToolkit.params.(state_list[d, :]))
            for i in 1:num_state
                state_list[d, i] = HMMToolkit.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty = penalty, pen_params_jk = pen_params[d][i])
                # Svi has not changed since E-step
            end

            ll_em_np = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
            ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter), updating dim $(d): $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
                if s == "-"
                    params_new = (x -> round.(x, digits = 4)).(HMMToolkit.params.(state_list[d, :]))
                    @info(
                        "Intended update of params: $(params_old) ->  $(params_new)"
                    )
                end
            end
            ll_em_temp = ll_em
            GC.safepoint()
        end

        GC.safepoint()
        
    end # iter
    
    converge = (0 <= ll_em - ll_em_old <= ϵ)

    num_params = _count_π(π_list) + _count_Q_mat(Q_mat) + _count_params(state_list)
    AIC = -2.0 * ll_em_np + 2 * num_params
    BIC = -2.0 * ll_em_np + log(nrow(df)) * num_params

    GC.safepoint()

    return (Q_mat_fit = Q_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, 
            ll = ll_em, ll_np = ll_em_np,
            AIC = AIC, BIC = BIC)

end