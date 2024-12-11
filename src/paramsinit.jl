
_default_expert_continuous = [
    GammaExpert()
    LogNormalExpert()
    ZILogNormalExpert()
    ZIGammaExpert()
]

_default_expert_real = [
    LaplaceExpert()
    NormalExpert()
    VonMisesExpert()
]

function _params_init_switch(Y_j, type_j)
    if type_j == "continuous"
        return params_init.(Ref(Y_j), _default_expert_continuous)
    elseif type_j == "discrete"
        return params_init.(Ref(Y_j), _default_expert_discrete)
    elseif type_j == "real"
        return params_init.(Ref(Y_j), _default_expert_real)
    else
        error("Invalid specification of distribution types.")
    end
end

function _sample_random_init(params_init)
    return vcat(
        [
            hcat(
                [
                    params_init[j][lb][rand(
                        Distributions.Categorical(
                            fill(1 / length(params_init[j][lb]), length(params_init[j][lb]))
                        ),
                        1,
                    )]
                    for lb in 1:length(params_init[1])
                ]...,
            )
            for j in 1:length(params_init)
        ]...,
    )
end


function cluster_responses(Y, n_cluster)

    Y = Array(Y)
    Y_std = Array{Float64}(Base.copy(Y))
    num_dim = size(Y)[2]

    for d in 1:num_dim
        Y_std[:, d] = (Y[:, d] .- mean(skipmissing(Y[:, d]))) ./ sqrt(var(skipmissing(Y[:, d])))
    end

    Y_std = Array{Float64}(coalesce.(Y_std, 0)) # required to not have inexact error during kmeans
    tmp = kmeans(Array(Y_std'), n_cluster)

    return (groups = assignments(tmp), prop = counts(tmp) ./ size(Y)[1])
end



function CTHMM_cmm_init_exact(Y, num_state, type; block_size = nothing)

    num_dim = size(Y)[2]
    label, prop = cluster_responses(Y, num_state)

    # random initialization of Q_mat
    A = rand(num_state, num_state)
    Q_mat_init = fill(0.0, num_state, num_state)

    if(!isnothing(block_size))
        num_block = length(block_size)
        for i in 1:num_block
            row = sum(block_size[1:i])
            start_idx = row - block_size[i] + 1
            Q_mat_init[(start_idx):(row - 1), (start_idx):(row)] = A[(start_idx):(row - 1), (start_idx):(row)]
            Q_mat_init[row, (start_idx):(row)] .= 1
        end
    else
        Q_mat_init = A
        Q_mat_init[num_state, :] .= 1
    end

    for i = 1:num_state
        Q_mat_init[i, i] = -(sum(Q_mat_init[i, :]) - Q_mat_init[i, i])    # sum the qij terms
    end

    if(isnothing(block_size))
        π_list_init = exp(Q_mat_init * 10000)[1, :]
    else
        tmp = sum(exp(Q_mat_init * 10000), dims = 1) ./ block_size[1]
        π_list_init = tmp ./ sum(tmp)
    end

    # summary statistics
    zero_y = [
        hcat(
            [
                sum(Y[label .== lb, j] .== 0.0) / length(Y[label .== lb, j]) for
                lb in unique(label)
            ]...,
        ) for j in 1:num_dim
    ]
    mean_y_pos = [
        hcat(
            [mean(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    mean_y = [
        hcat(
            [mean(Y[label .== lb, j]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    var_y_pos = [
        hcat(
            [var(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    var_y = [
        hcat(
            [var(Y[label .== lb, j]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    skewness_y_pos = [
        hcat(
            [skewness(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    kurtosis_y_pos = [
        hcat(
            [kurtosis(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]

    # initialize component distributions
    params_init = [
        [_params_init_switch(Y[label .== lb, j], type[j]) for lb in unique(label)] for
        j in 1:num_dim
    ]

    return (π_list_init = π_list_init, Q_mat_init = Q_mat_init, 
        params_init = params_init,
        zero_y = zero_y,
        mean_y_pos = mean_y_pos, mean_y = mean_y,
        var_y_pos = var_y_pos, var_y = var_y,
        skewness_y_pos = skewness_y_pos, kurtosis_y_pos = kurtosis_y_pos)
end



function DTHMM_cmm_init_exact(Y, num_state, type; random_init = true, block_size = nothing)

    num_dim = size(Y)[2]
    label, prop = cluster_responses(Y, num_state)

    # random initialization of P_mat
    if (random_init)
        P_mat_init = rand(1:10, num_state, num_state)
        P_mat_init = P_mat_init ./ sum(P_mat_init, dims = 2)

        π_list_init = rand(1:10, num_state)
        π_list_init = π_list_init ./ sum(π_list_init)
    else
        P_mat_init = fill(1/num_state, num_state, num_state)
        π_list_init = fill(1/num_state, num_state)
    end

    if(!isnothing(block_size))
        num_block = length(block_size)
        tmp = zeros(num_state, num_state)
        for i in 1:num_block
            row = sum(block_size[1:i])
            start_idx = row - block_size[i] + 1
            tmp[(start_idx):(row), (start_idx):(row)] = P_mat_init[(start_idx):(row), (start_idx):(row)]
        end
        P_mat_init = tmp ./ sum(tmp, dims = 2)
    end

    # summary statistics
    zero_y = [
        hcat(
            [
                sum(Y[label .== lb, j] .== 0.0) / length(Y[label .== lb, j]) for
                lb in unique(label)
            ]...,
        ) for j in 1:num_dim
    ]
    mean_y_pos = [
        hcat(
            [mean(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    mean_y = [
        hcat(
            [mean(Y[label .== lb, j]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    var_y_pos = [
        hcat(
            [var(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    var_y = [
        hcat(
            [var(Y[label .== lb, j]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    skewness_y_pos = [
        hcat(
            [skewness(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]
    kurtosis_y_pos = [
        hcat(
            [kurtosis(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:num_dim
    ]

    # initialize component distributions
    params_init = [
        [_params_init_switch(Y[label .== lb, j], type[j]) for lb in unique(label)] for
        j in 1:num_dim
    ]

    return (π_list_init = π_list_init, P_mat_init = P_mat_init, 
        params_init = params_init,
        zero_y = zero_y,
        mean_y_pos = mean_y_pos, mean_y = mean_y,
        var_y_pos = var_y_pos, var_y = var_y,
        skewness_y_pos = skewness_y_pos, kurtosis_y_pos = kurtosis_y_pos)
end



"""
    CTHMM_cmm_init(Y, num_state, type; block_size = nothing, n_random = 5)

Initialize a CTHMM model using the Clustered Method of Moments (CMM).

# Arguments
- `Y`: A matrix of response.
- `num_state`: Integer. Number of latent states.
- `type`: A vector of either `continuous`, `discrete` or `real`, indicating the type of response by dimension.

# Optional Arguments
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `n_random`: Integer. Number of randomized initializations. Default to 5.

# Return Values
- `π_list_init`: Initialization of initial state probabilities `π_list`.
- `Q_mat_init`: Initialization of transition rate matrix `Q_mat`.
- `params_init`: Initializations of state dependent distributions. It is a three-dimensional vector. 
    For example, `params_init[1][2]` initializes the 1st dimension of `Y` using the 2nd latent class,
    which is a vector of potential expert functions to choose from.
- `zero_y`: Proportion of zeros in observed `Y`.
- `mean_y_pos`: Mean of positive observations in `Y`.
- `mean_y`: Mean of observations in `Y`.
- `var_y_pos`: Variance of positive observations in `Y`.
- `var_y`: Variance of observations in `Y`.
- `skewness_y_pos`: Skewness of positive observations in `Y`.
- `kurtosis_y_pos`: Kurtosis of positive observations in `Y`.
- `random_init`: A list of `n_random` randomized initializations chosen from `params_init`.
"""

function CTHMM_cmm_init(Y, num_state, type; block_size = nothing, n_random = 5)

    tmp = cmm_init_exact(Y, num_state, type; block_size = block_size)

    if n_random >= 1
        random_init = [_sample_random_init(tmp.params_init) for i in 1:n_random]
    else
        random_init = nothing
    end

    return (π_list_init = tmp.π_list_init, Q_mat_init = tmp.Q_mat_init, 
        params_init = tmp.params_init,
        zero_y = tmp.zero_y,
        mean_y_pos = tmp.mean_y_pos, mean_y = tmp.mean_y,
        var_y_pos = tmp.var_y_pos, var_y = tmp.var_y,
        skewness_y_pos = tmp.skewness_y_pos, kurtosis_y_pos = tmp.kurtosis_y_pos, 
        random_init = random_init)
end



"""
    DTHMM_cmm_init(Y, num_state, type; block_size = nothing, n_random = 5)

Initialize a DTHMM model using the Clustered Method of Moments (CMM).

# Arguments
- `Y`: A matrix of response.
- `num_state`: Integer. Number of latent states.
- `type`: A vector of either `continuous`, `discrete` or `real`, indicating the type of response by dimension.

# Optional Arguments
- `block_size`: A vector specifying the structure of the trasition probability matrix. Default to nothing (not a block matrix).
- `random_init`: `true`(default) or `false`, indicating whether `π_list` and `P_mat` are initialized randomly (default) or uniformly.
- `n_random`: Integer. Number of randomized initializations. 

# Return Values
- `π_list_init`: Initialization of initial state probabilities `π_list`.
- `P_mat_init`: Initialization of transition probability matrix `P_mat`.
- `params_init`: Initializations of state dependent distributions. It is a three-dimensional vector. 
    For example, `params_init[1][2]` initializes the 1st dimension of `Y` using the 2nd latent class,
    which is a vector of potential expert functions to choose from.
- `zero_y`: Proportion of zeros in observed `Y`.
- `mean_y_pos`: Mean of positive observations in `Y`.
- `mean_y`: Mean of observations in `Y`.
- `var_y_pos`: Variance of positive observations in `Y`.
- `var_y`: Variance of observations in `Y`.
- `skewness_y_pos`: Skewness of positive observations in `Y`.
- `kurtosis_y_pos`: Kurtosis of positive observations in `Y`.
- `random_init`: A list of `n_random` randomized initializations chosen from `params_init`.
"""

function DTHMM_cmm_init(Y, num_state, type; random_init = true, block_size = nothing, n_random = 5)

    tmp = DTHMM_cmm_init_exact(Y, num_state, type; random_init = true, block_size = block_size)

    if n_random >= 1
        random_init = [_sample_random_init(tmp.params_init) for i in 1:n_random]
    else
        random_init = nothing
    end

    return (π_list_init = tmp.π_list_init, P_mat_init = tmp.P_mat_init, 
        params_init = tmp.params_init,
        zero_y = tmp.zero_y,
        mean_y_pos = tmp.mean_y_pos, mean_y = tmp.mean_y,
        var_y_pos = tmp.var_y_pos, var_y = tmp.var_y,
        skewness_y_pos = tmp.skewness_y_pos, kurtosis_y_pos = tmp.kurtosis_y_pos, 
        random_init = random_init)
end