function build_cov_Q(num_state, α, X; check_args = true)
    # size(α, 2) is the number of covariates C, including the intercept
    # X is a 1 by C matrix holding the a specific covariate combination
    check_args && @check_args(build_cov_Q, size(α, 1) == num_state*(num_state - 1))
    Q = zeros(num_state, num_state)
    ax = X * α'
    exp_ax = exp.(ax)

    k = 0
    for i in 1:num_state    # fill by row    
        for j in 1:num_state
            if i != j
                k = k + 1
                Q[i, j] = exp_ax[k]
            end
        end
        Q[i, i] = -sum(Q[i, :])
    end

    return Q
end

