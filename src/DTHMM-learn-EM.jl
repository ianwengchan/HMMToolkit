"""
    DTHMM_learn_EM(df, response_list, P_mat_init, π_list_init, state_list_init;
        ϵ = 1e-03, max_iter = 200, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

Fit a DTHMM model via EM algorithm.

# Arguments
- `df`: Dataframe with (multiple) time series.
- `response_list`: List of responses to consider.
- `P_mat_init`: Initial estimate of the transition probability matrix `P_mat`.
- `π_list_init`: Initial estimate of the initial state probabilities `π_list`.
- `state_list_init`: Initial estimate of the state dependent distributions `state_list`.

# Optional Arguments
- `ϵ`: Stopping criterion on loglikelihood, stop when the increment is less than ϵ. Default to 0.001.
- `max_iter`: Maximum number of iterations of the EM algorithm. Default to 200.
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `print_steps`: Logs parameter updates every (print_steps) iterations of the EM algorithm. Default to 1.
- `penalty`: `true` or `false`(default), indicating whether penalty is imposed on the magnitude of state dependent distribution parameters.
- `pen_params`: an array of penalty term on the magnitude of state dependent distribution parameters.
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `group_by_col`: List of columns to group the dataframe. If nothing is provided, group by "ID" by default.

# Return Values
- `P_mat_fit`: Fitted transition probability matrix `P_mat`.
- `π_list_fit`: Fitted initial state probabilities `π_list`.
- `state_list_fit`: Fitted state dependent distributions `state_list`.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting procedure.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np``: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""

function DTHMM_learn_EM(df, response_list, P_mat_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, soft_decode = 1, print_steps = 1, penalty = false, pen_params = nothing, block_size = nothing, group_by_col = nothing)

    ## precomputation before iteration
    # only once in the whole EM:
    num_state = size(P_mat_init, 1)
    num_dim = size(state_list_init, 1)
    
    # start EM iteration
    P_mat = Base.copy(P_mat_init)
    π_list = Base.copy(π_list_init)
    state_list = Base.copy(state_list_init)

    # initialize pen_params if not provided OR penalty is false
    if penalty == false
        pen_params = [HMMToolkit.no_penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    elseif isnothing(pen_params)
        pen_params = [HMMToolkit.penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    end

    ll_em_old = -Inf
    ll_em_np = DTHMM_batch_decode_for_subjects(soft_decode, df, response_list, P_mat, π_list, state_list)
    ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
    ll_em = ll_em_np + ll_em_penalty
    iter = 0

    GC.safepoint()
    
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
            
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df
        ll_em_temp, Eij_all = DTHMM_batch_decode_Eij_and_append_Svi_for_subjects(soft_decode, df, response_list, P_mat, π_list, state_list)

        GC.safepoint()
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em_np = DTHMM_batch_decode_for_subjects(soft_decode, df, response_list, P_mat, π_list, state_list)
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

        ## part 1b: updating P_mat

        P_old = copy(P_mat)
        P_mat = DTHMM_learn_update_P_mat(Eij_all; block_size = block_size)

        GC.safepoint()

        ll_em_np = DTHMM_batch_decode_for_subjects(soft_decode, df, response_list, P_mat, π_list, state_list)
        ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
        ll_em = ll_em_np + ll_em_penalty

        s = ll_em - ll_em_temp > 0 ? "+" : "-"
        pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
        if (print_steps > 0) & (iter % print_steps == 0)
            @info(
                "Iteration $(iter), updating P: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
            )
        end
        ll_em_temp = ll_em
        GC.safepoint()

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            params_old = (x -> round.(x, digits = 4)).(HMMToolkit.params.(state_list[d, :]))
            for i in 1:num_state
                state_list[d, i] = HMMToolkit.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty=penalty, pen_params_jk = pen_params[d][i])
                # Svi has not been changed since E-step
            end

            ll_em_np = DTHMM_batch_decode_for_subjects(soft_decode, df, response_list, P_mat, π_list, state_list)
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

    num_params = _count_π(π_list) + _count_P_mat(P_mat) + _count_params(state_list)
    AIC = -2.0 * ll_em_np + 2 * num_params
    BIC = -2.0 * ll_em_np + log(nrow(df)) * num_params

    GC.safepoint()

    return (P_mat_fit = P_mat, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, 
            ll = ll_em, ll_np = ll_em_np,
            AIC = AIC, BIC = BIC)

end