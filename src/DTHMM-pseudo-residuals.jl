"""
    DTHMM_ordinary_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list;
        keep_last = 1, ZI_list = nothing)

Computes the uniform or normal ordinary pseudo residuals of a single time series given a fitted DTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `response_list`: List of responses to consider.
- `data_emiss_prob_list_total`: An array of data emission probabilities; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `data_emiss_cdf_list_separate`: An array of (num_dim) data emission cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_separate` function (for the time series).
- `P_mat`: Fitted transition probability matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.
- `ZI_list`: A list of zero-inflated responses. Default to nothing.

# Return Values
- A (len_time_series)-by-(num_dim) array of ordinary pseudo-residuals.
"""

function DTHMM_ordinary_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list; keep_last = 1, ZI_list = nothing)

    # compute the probability conditioned on entire trip but the specific observation - ordinary pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)
    
    # ZI_list is a list

    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ALPHA_denom = zeros(len_time_series, num_state)
    ALPHA_nume = [zeros(len_time_series, num_state) for i in 1:num_dim]
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    # prepare numerator and denominator components for time 1
    ALPHA_denom[1, :] = transpose(π_list)  # only

    @threads for d in 1:num_dim
        data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
        ALPHA_nume[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_list_d[1, :])
    end

    ## init alpha for time 1
    ALPHA[1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_prob_list_total[1, :])

    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    
    for v = 2:len_time_series

        ALPHA_denom[v, :] = transpose(ALPHA[v-1, :]) * P_mat  # only

        @threads for d in 1:num_dim
            data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
            ALPHA_nume[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_list_d[v, :])
        end

        ALPHA[v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_prob_list_total[v, :])

        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
        
    end # v

    GC.safepoint()
    
    ## compute beta()
    BETA = zeros(len_time_series, num_state)

    # init beta()
    BETA[len_time_series, :] .= C[len_time_series]
    
    for v = (len_time_series-1):(-1):1
        
        BETA[v, :] = P_mat * Diagonal(data_emiss_prob_list_total[v+1, :]) * BETA[v+1, :]
        BETA[v, :] = BETA[v, :] .* C[v]
    end # v

    GC.safepoint()

    ## group denominator
    denominator = sum(ALPHA_denom .* BETA, dims = 2)

    residuals = zeros(len_time_series, num_dim)
    @threads for d in 1:num_dim
        residuals[:, d] = sum(ALPHA_nume[d] .* BETA, dims = 2) ./ denominator
    end

    residuals = [maximum([0, minimum([1, element])]) for element in residuals]

    # the pseudo-residuals are not uniform for ZI-distributions; adjustment for the point-mass at zero
    if (!isnothing(ZI_list)) # ZI exists
        @threads for d = 1:num_dim
            if (response_list[d] in ZI_list) # if dim d is ZI, do adjustment
                runif = rand(length(residuals[:, d]))
                residuals[:, d] = residuals[:, d] .* [x > 0 ? 1 : (1 - y) for (x, y) in zip(seq_df[1:end, response_list[d]], runif)]
            end
        end
    end
    
    if (uniform == 0) # normal
        residuals = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), residuals)
    end
    
    if (keep_last == 0) # remove last forecast
        residuals = residuals[1:(end-1), :]
    end

    return residuals
    
end



"""
    CTHMM_forecast_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list;
        keep_last = 1, ZI_list = nothing)

Computes the uniform or normal forecast pseudo residuals of a single time series given a fitted DTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `response_list`: List of responses to consider.
- `data_emiss_prob_list_total`: An array of data emission probabilities; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `data_emiss_cdf_list_separate`: An array of (num_dim) data emission cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_separate` function (for the time series).
- `P_mat`: Fitted transition probability matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.
- `ZI_list`: A list of zero-inflated responses. Default to nothing.

# Return Values
- A (len_time_series - 1)-by-(num_dim) array of forecast pseudo-residuals.
"""

function DTHMM_forecast_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list; keep_last = 1, ZI_list = nothing)
    
    # compute the posterior probability conditioned on entire history - forecast pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)
    
    # ZI_list is a list
    
    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)
    
    GC.safepoint()
    
    forecast = zeros((len_time_series - 1), num_dim) # store the forecast pseudo-residuals for each dimension
    
    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ## for each dimension d, ALPHA_nume[2:end] is also the forecast pseudo-residual!
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    tmp = transpose(π_list)

    ## init alpha for time 1
    ALPHA[1, :] = transpose(tmp) * Diagonal(data_emiss_prob_list_total[1, :])
    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    # no forecast pseudo-residuals at time 1


    for v = 2:len_time_series

        tmp = transpose(ALPHA[v-1, :]) * P_mat  # only

        @threads for d in 1:num_dim
            data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
            forecast[v-1, d] = sum(transpose(tmp) * Diagonal(data_emiss_cdf_list_d[v, :]))
        end

        ALPHA[v, :] = transpose(tmp) * Diagonal(data_emiss_prob_list_total[v, :])

        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
        
    end # v
    
    GC.safepoint()
    
    # the forecast pseudo-residuals are not uniform for ZI-distributions; adjustment for the point-mass at zero
    if (!isnothing(ZI_list)) # ZI exists
        for d = 1:num_dim
            if (response_list[d] in ZI_list) # if dim d is ZI, do adjustment
                runif = rand(length(forecast[:, d]))
                forecast[:, d] = forecast[:, d] .* [x > 0 ? 1 : (1 - y) for (x, y) in zip(seq_df[2:end, response_list[d]], runif)]
            end
        end
    end
    
    if (uniform == 0) # normal
        forecast = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), forecast)
    end
    
    if (keep_last == 0) # remove last forecast
        forecast = forecast[1:(end-1), :]
    end

    return forecast
    
end



"""
    DTHMM_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list;
        keep_last = 1, ZI_list = nothing)

Computes both the uniform or normal ordinary and forecast pseudo residuals of a single time series given a fitted DTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `response_list`: List of responses to consider.
- `data_emiss_prob_list_total`: An array of data emission probabilities; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `data_emiss_cdf_list_separate`: An array of (num_dim) data emission cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_separate` function (for the time series).
- `P_mat`: Fitted transition probability matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.
- `ZI_list`: A list of zero-inflated responses. Default to nothing.

# Return Values
- `ordinary_residuals`: A (len_time_series)-by-(num_dim) array of ordinary pseudo-residuals.
- `forecast_residuals`: A (len_time_series - 1)-by-(num_dim) array of forecast pseudo-residuals.
"""

function DTHMM_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list; keep_last = 1, ZI_list = nothing)
    
    # compute both the ordinary and forecast pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)
    
    # ZI_list is a list

    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ALPHA_denom = zeros(len_time_series, num_state)
    ALPHA_nume = [zeros(len_time_series, num_state) for i in 1:num_dim]
    ## for each dimension d, ALPHA_nume[2:end] is also the forecast pseudo-residual!
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    # prepare numerator and denominator components for time 1
    ALPHA_denom[1, :] = transpose(π_list)  # only

    for d in 1:num_dim
        data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
        ALPHA_nume[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_list_d[1, :])
    end

    ## init alpha for time 1
    ALPHA[1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_prob_list_total[1, :])

    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    
    for v = 2:len_time_series

        ALPHA_denom[v, :] = transpose(ALPHA[v-1, :]) * P_mat  # only

        @threads for d in 1:num_dim
            data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
            ALPHA_nume[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_list_d[v, :])
        end

        ALPHA[v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_prob_list_total[v, :])

        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
        
    end # v

    GC.safepoint()
    
    ## compute beta()
    BETA = zeros(len_time_series, num_state)

    # init beta()
    BETA[len_time_series, :] .= C[len_time_series]
    
    for v = (len_time_series-1):(-1):1
        
        BETA[v, :] = P_mat * Diagonal(data_emiss_prob_list_total[v+1, :]) * BETA[v+1, :]
        BETA[v, :] = BETA[v, :] .* C[v]
    end # v

    GC.safepoint()

    ## group denominator
    denominator = sum(ALPHA_denom .* BETA, dims = 2)

    residuals = zeros(len_time_series, num_dim)
    for d in 1:num_dim
        residuals[:, d] = sum(ALPHA_nume[d] .* BETA, dims = 2) ./ denominator
    end

    residuals = [maximum([0, minimum([1, element])]) for element in residuals]

    forecast = zeros(len_time_series-1, num_dim)
    for d in 1:num_dim
        forecast[:, d] = sum(ALPHA_nume[d][2:end, :], dims = 2)
    end

    forecast = [maximum([0, minimum([1, element])]) for element in forecast]

    
    # the pseudo-residuals are not uniform for ZI-distributions; adjustment for the point-mass at zero
    if (!isnothing(ZI_list)) # ZI exists
        @threads for d = 1:num_dim
            if (response_list[d] in ZI_list) # if dim d is ZI, do adjustment
                runif = rand(length(forecast[:, d]))
                forecast[:, d] = forecast[:, d] .* [x > 0 ? 1 : (1 - y) for (x, y) in zip(seq_df[2:end, response_list[d]], runif)]
                
                runif = rand(length(residuals[:, d]))
                residuals[:, d] = residuals[:, d] .* [x > 0 ? 1 : (1 - y) for (x, y) in zip(seq_df[1:end, response_list[d]], runif)]
            end
        end
    end
    
    if (uniform == 0) # normal
        forecast = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), forecast)
        residuals = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), residuals)
    end
    
    if (keep_last == 0) # remove last forecast
        forecast = forecast[1:(end-1), :]
        residuals = residuals[1:(end-1), :]
    end

    return (ordinary_residuals = residuals, forecast_residuals = forecast)
    
end



"""
    DTHMM_batch_pseudo_residuals(uniform, df, response_list, P_mat, π_list, state_list;
        keep_last = 1, ZI_list = nothing, group_by_col = nothing)

Computes both the uniform or normal ordinary and forecast pseudo residuals of (multiple) time series in batches given a fitted DTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `P_mat`: Fitted transition probability matrix.
- `π_list`: Fitted initial state probabilities.
- `state_list`: Fitted state dependent distributions.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.
- `ZI_list`: A list of zero-inflated responses. Default to nothing.

# Return Values
- `ordinary_list`: A list of g ordinary pseudo-residual array for g time series.
- `forecast_list`: A list of g forecast pseudo-residual array for g time series.
"""

function DTHMM_batch_pseudo_residuals(uniform, df, response_list, P_mat, π_list, state_list; ϵ = 0, keep_last = 1, ZI_list = nothing)
    
    # batch compute both the ordinary and forecast pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)

    ## precomputation
    obs_seq_emiss_list_total = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list)
    obs_seq_cdf_list_separate = CTHMM_precompute_batch_data_emission_cdf_separate(df, response_list, state_list)

    num_state = size(P_mat, 1)

    group_df = groupby(df, :ID)
    num_time_series = size(group_df, 1)

    ordinary_list = Array{Array{Float64}}(undef, num_time_series)
    forecast_list = Array{Array{Float64}}(undef, num_time_series)
    
    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = group_df[g]

        data_emiss_prob_list_total = obs_seq_emiss_list_total[g][1]
        data_emiss_cdf_list_separate = obs_seq_cdf_list_separate[g]
        ordinary_list[g], forecast_list[g] = DTHMM_pseudo_residuals(uniform, seq_df, response_list, data_emiss_prob_list_total, data_emiss_cdf_list_separate, P_mat, π_list; ϵ, keep_last, ZI_list)

        GC.safepoint()
        
    end # g

    GC.safepoint()

    return (ordinary_list = ordinary_list, forecast_list = forecast_list)
    
end




"""
    DTHMM_batch_anomaly_indices(df, response_list, P_mat, π_list, state_list;
        keep_last = 1, ZI_list = nothing, group_by_col = nothing)

Computes the anomaly indices of (multiple) time series in batches given a fitted DTHMM.

# Arguments
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `P_mat`: Fitted transition probability matrix.
- `π_list`: Fitted initial state probabilities.
- `state_list`: Fitted state dependent distributions.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.
- `ZI_list`: A list of zero-inflated responses. Default to nothing.

# Return Values
- A g-by-(num_dim) array storing the anomaly indices of the g time series, for each of the (num_dim) response dimensions.
"""

function DTHMM_batch_anomaly_indices(df, response_list, P_mat, π_list, state_list; keep_last = 1, ZI_list = nothing, group_by_col = nothing)

    # uniform = 0 for normal pseudo-residuals
    ordinary_list, forecast_list = DTHMM_batch_pseudo_residuals(0, df, response_list, P_mat, π_list, state_list; 
                                                                    keep_last = keep_last, ZI_list = ZI_list, group_by_col = group_by_col)
    
    anomaly_indices = CTHMM_anomaly_indices(forecast_list)

    return anomaly_indices

end