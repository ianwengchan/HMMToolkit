function CTHMM_learn_cov_EM(df, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init;
    ϵ = 1e-03, max_iter = 200, α_max_iter = 5, soft_decode = 1, print_steps = 1, penalty = true, pen_params = nothing)

    ## precomputation before iteration
    # only once in the whole EM:
    num_state = size(π_list_init, 1)
    num_dim = size(state_list_init, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    distinct_time_list = Array{Vector{Int64}}(undef, num_subject) # the distinct_time_list depends on the subject
    @threads for n = 1:num_subject
        distinct_time_list[n] = CTHMM_precompute_distinct_time_list(group_df[n].time_interval)
        GC.safepoint()
    end

    GC.safepoint()

    # for i = 1:num_state
    #     subject_df[:, string("tau", i)] = missings(Float64, num_subject)
    #     for j = 1:num_state
    #         subject_df[:, string("N", i, j)] = missings(Float64, num_subject)
    #     end
    # end

    
    # start EM iteration
    α = Base.copy(α_init)
    X = Matrix(subject_df[!, covariate_list])   # same for all elements of Q and across iterations
    
    π_list = Base.copy(π_list_init)
    state_list = Base.copy(state_list_init)

    # initialize pen_params if not provided OR penalty is false
    if penalty == false
        pen_params = [HMMToolkit.no_penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    elseif isnothing(pen_params)
        pen_params = [HMMToolkit.penalty_init.(state_list[k, :]) for k in 1:size(state_list)[1]]
    end

    ll_em_old = -Inf
    ll_em_np = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
    ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
    ll_em = ll_em_np + ll_em_penalty
    iter = 0

    GC.safepoint()
    
    while (abs(ll_em - ll_em_old) > ϵ) && (iter < max_iter)
        
        ## add counter
        iter = iter + 1
        ll_em_old = ll_em
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df; need to compute Etij separately for each subject
        ll_em_temp, Etij_list = CTHMM_batch_decode_Etij_and_append_Svi_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)

        GC.safepoint()
        
        ## M-step, with last estimated parameters
        ## part 1a: learning initial state probabilities π_list
        for i in 1:num_state
            π_list[i] = sum(df.start .* df[:, string("Sv", i)])
        end
        π_list = π_list ./ sum(π_list)

        ll_em_np = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
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

        ## part 1b: learning α using distinct time grouping

        α_old = copy(α) .- Inf

        α_iter = 0
        while (α_iter < α_max_iter) && (sum((α - α_old) .^ 2) > 1e-10)
            α_iter = α_iter + 1
            α_old = copy(α)

            # have to find Nij_mat and taui_list for each subject
            subject_df = CTHMM_learn_cov_nij_taui(num_state, num_subject, subject_df, covariate_list, distinct_time_list, α, Etij_list)
            
            @threads for i in 1:(num_state-1)    # fill by row
                tau = subject_df[!, string("tau", i)]
                w = coalesce.(tau, 0.0)
                @threads for j in 1:num_state
                    if i != j
                        # k = k + 1
                        k = (num_state * i) - (num_state - j)  - (i - (i >= j ? 1 : 0))
                        y = subject_df[!, string("N", i, j)] ./ tau
                        α[k, :] = coef(glm(X, y, Gamma(), LogLink(), wts = w))
                        # println(α)
                        GC.safepoint()
                    end
                    GC.safepoint()
                end
                GC.safepoint()
            end

            GC.safepoint()
            
            ll_em_np = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
            ll_em_penalty = penalty ? penalty_params(state_list, pen_params) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if (print_steps > 0) & (iter % print_steps == 0)
                @info(
                    "Iteration $(iter) sub $(α_iter), updating α: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
                )
                @info(
                    "New α: $(round.(α, digits = 5))"
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
                state_list[d, i] = HMMToolkit.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)]; penalty=penalty, pen_params_jk = pen_params[d][i])
                # Svi has not been changed since E-step
            end
            ll_em_np = CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list)
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

    GC.safepoint()

    return (α_fit = α, π_list_fit = π_list, state_list_fit = state_list,
            converge = converge, iter = iter, ll = ll_em, ll_np = ll_em_np)

end