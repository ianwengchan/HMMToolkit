"""
    randomized_pit(U_upper, U_lower)

Given upper and lower PIT values, return randomized PIT values:
    U = U_lower + V * (U_upper - U_lower),
where V ~ Unif(0,1) independently.

If U_upper == U_lower, this reduces to the ordinary PIT.
"""
function randomized_pit(U_upper::AbstractMatrix, U_lower::AbstractMatrix)

    n, num_dim = size(U_upper)
    U = Array{Float64}(undef, n, num_dim)

    @threads for d in 1:num_dim
        runif = rand(n)
        U[:, d] = U_lower[:, d] .+ runif .* (U_upper[:, d] .- U_lower[:, d])
        GC.safepoint()
    end

    U .= clamp.(U, 0.0, 1.0)

    return U
end


# Legacy low-level calls supply response names and optional ZI_list instead of
# lower-CDF matrices. Keep their zero-mass handling; batch calls infer it from experts.
function _legacy_residual_cdf_lower(seq_df, response_list, cdf_upper, ZI_list)
    cdf_lower = deepcopy(cdf_upper)
    if !isnothing(ZI_list)
        for d in eachindex(response_list)
            if response_list[d] in ZI_list
                for (v, y) in enumerate(seq_df[!, response_list[d]])
                    if !ismissing(y) && y == 0
                        cdf_lower[d][v, :] .= 0.0
                    end
                end
            end
        end
    end
    return cdf_lower
end




"""
    CTHMM_ordinary_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list;
        keep_last = 1)

Computes the uniform or normal ordinary pseudo residuals of a single time series given a fitted CTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `data_emiss_prob_list_total`: An array of data emission probabilities; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `data_emiss_cdf_list_separate`: An array of (num_dim) data emission cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_separate` function (for the time series).
- `data_emiss_cdf_lower_list_separate`: An array of (num_dim) data emission lower cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_lower_separate` function (for the time series).
- `Q_mat`: Fitted transition rate matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.

# Return Values
- A (len_time_series)-by-(num_dim) array of ordinary pseudo-residuals.
"""

function CTHMM_ordinary_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list; keep_last = 1)

    # compute the probability conditioned on entire trip but the specific observation - ordinary pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)

    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    ## precomputing of Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end

    GC.safepoint()

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ALPHA_denom = zeros(len_time_series, num_state)
    # ALPHA_nume = [zeros(len_time_series, num_state) for i in 1:num_dim]
    ALPHA_nume_upper = [zeros(len_time_series, num_state) for i in 1:num_dim]
    ALPHA_nume_lower = [zeros(len_time_series, num_state) for i in 1:num_dim]
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    # prepare numerator and denominator components for time 1
    ALPHA_denom[1, :] = transpose(π_list)  # only

    # @threads for d in 1:num_dim
    #     data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
    #     ALPHA_nume[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_list_d[1, :])
    # end

    @threads for d in 1:num_dim
        data_emiss_cdf_upper_d = data_emiss_cdf_list_separate[d]
        data_emiss_cdf_lower_d = data_emiss_cdf_lower_list_separate[d]

        ALPHA_nume_upper[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_upper_d[1, :])
        ALPHA_nume_lower[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_lower_d[1, :])
    end

    ## init alpha for time 1
    ALPHA[1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_prob_list_total[1, :])

    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    
    for v = 2:len_time_series

        ALPHA_denom[v, :] = transpose(ALPHA[v-1, :]) * Pt_list[v-1]  # only

        # @threads for d in 1:num_dim
        #     data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
        #     ALPHA_nume[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_list_d[v, :])
        # end
        @threads for d in 1:num_dim
            data_emiss_cdf_upper_d = data_emiss_cdf_list_separate[d]
            data_emiss_cdf_lower_d = data_emiss_cdf_lower_list_separate[d]

            ALPHA_nume_upper[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_upper_d[v, :])
            ALPHA_nume_lower[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_lower_d[v, :])
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
        
        BETA[v, :] = Pt_list[v] * Diagonal(data_emiss_prob_list_total[v+1, :]) * BETA[v+1, :]
        BETA[v, :] = BETA[v, :] .* C[v]
    end # v

    GC.safepoint()

    ## group denominator
    denominator = sum(ALPHA_denom .* BETA, dims = 2)

    # residuals = zeros(len_time_series, num_dim)
    # @threads for d in 1:num_dim
    #     residuals[:, d] = sum(ALPHA_nume[d] .* BETA, dims = 2) ./ denominator
    # end

    residuals_upper = zeros(len_time_series, num_dim)
    residuals_lower = zeros(len_time_series, num_dim)

    @threads for d in 1:num_dim
        residuals_upper[:, d] = vec(sum(ALPHA_nume_upper[d] .* BETA, dims = 2) ./ denominator)
        residuals_lower[:, d] = vec(sum(ALPHA_nume_lower[d] .* BETA, dims = 2) ./ denominator)
    end

    # residuals = [maximum([0, minimum([1, element])]) for element in residuals]

    residuals_upper = clamp.(residuals_upper, 0.0, 1.0)
    residuals_lower = clamp.(residuals_lower, 0.0, 1.0)

    residuals = randomized_pit(residuals_upper, residuals_lower)
    
    if (uniform == 0) # normal
        residuals = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), residuals)
    end
    
    if (keep_last == 0) # remove last residual
        residuals = residuals[1:(end-1), :]
    end

    return residuals
    
end



"""
    CTHMM_forecast_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list;
        keep_last = 1)

Computes the uniform or normal forecast pseudo residuals of a single time series given a fitted CTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `data_emiss_prob_list_total`: An array of data emission probabilities; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `data_emiss_cdf_list_separate`: An array of (num_dim) data emission cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_separate` function (for the time series).
- `data_emiss_cdf_lower_list_separate`: An array of (num_dim) data emission lower cdf; returned by `CTHMM_precompute_batch_data_emission_cdf_lower_separate` function (for the time series).
- `Q_mat`: Fitted transition rate matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.

# Return Values
- A (len_time_series - 1)-by-(num_dim) array of forecast pseudo-residuals.
"""

function CTHMM_forecast_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list; keep_last = 1)
    
    # compute the posterior probability conditioned on entire history - forecast pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)
    
    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
    
    ## precomputing of Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end
    
    GC.safepoint()
    
    # forecast = zeros((len_time_series - 1), num_dim) # store the forecast pseudo-residuals for each dimension
    forecast_upper = zeros((len_time_series - 1), num_dim)
    forecast_lower = zeros((len_time_series - 1), num_dim)
    
    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ## for each dimension d, ALPHA_nume[2:end] is also the forecast pseudo-residual!
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    tmp = transpose(π_list)

    ## init alpha for time 1
    ALPHA[1, :] = tmp * Diagonal(data_emiss_prob_list_total[1, :])
    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    # no forecast pseudo-residuals at time 1


    for v = 2:len_time_series

        tmp = transpose(ALPHA[v-1, :]) * Pt_list[v-1]  # only

        # @threads for d in 1:num_dim
        #     data_emiss_cdf_list_d = data_emiss_cdf_list_separate[d]
        #     forecast[v-1, d] = sum(transpose(tmp) * Diagonal(data_emiss_cdf_list_d[v, :]))
        # end

        @threads for d in 1:num_dim
            data_emiss_cdf_upper_d = data_emiss_cdf_list_separate[d]
            data_emiss_cdf_lower_d = data_emiss_cdf_lower_list_separate[d]

            forecast_upper[v-1, d] = sum(tmp * Diagonal(data_emiss_cdf_upper_d[v, :]))
            forecast_lower[v-1, d] = sum(tmp * Diagonal(data_emiss_cdf_lower_d[v, :]))
        end

        ALPHA[v, :] = tmp * Diagonal(data_emiss_prob_list_total[v, :])

        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
        
    end # v
    
    GC.safepoint()

    forecast_upper = clamp.(forecast_upper, 0.0, 1.0)
    forecast_lower = clamp.(forecast_lower, 0.0, 1.0)

    forecast = randomized_pit(forecast_upper, forecast_lower)
    
    if (uniform == 0) # normal
        forecast = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), forecast)
    end
    
    if (keep_last == 0) # remove last forecast
        forecast = forecast[1:(end-1), :]
    end

    return forecast
    
end



"""
    CTHMM_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list;
        keep_last = 1)

Computes both the uniform or normal ordinary and forecast pseudo residuals of a single time series given a fitted CTHMM, in one pass.

# Arguments
- `uniform`: 1 for uniform pseudo-residuals, normal otherwise.
- `seq_df`: Dataframe of a single time series.
- `data_emiss_prob_list_total`: Array of data emission probabilities.
- `data_emiss_cdf_list_separate`: Array of upper CDF values F(y) for each dimension.
- `data_emiss_cdf_lower_list_separate`: Array of lower CDF values F(y^-) for each dimension.
- `Q_mat`: Fitted transition rate matrix.
- `π_list`: Fitted initial state probabilities.

# Optional Arguments
- `keep_last`: 1 (default) if the last observation is kept, removed otherwise.

# Return Values
- `ordinary_residuals`: A (len_time_series)-by-(num_dim) array of ordinary pseudo-residuals.
- `forecast_residuals`: A (len_time_series - 1)-by-(num_dim) array of forecast pseudo-residuals.
"""
function CTHMM_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list; keep_last = 1)

    # compute both:
    # - ordinary pseudo-residuals: conditioned on the whole trip except the specific observation
    # - forecast pseudo-residuals: conditioned on the history only

    num_dim = size(data_emiss_cdf_list_separate, 1)
    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    ## precompute Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, len_time_series - 1)
    @threads for v = 1:(len_time_series - 1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end

    GC.safepoint()

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    ALPHA_denom = zeros(len_time_series, num_state)

    # for ordinary residuals
    ALPHA_nume_upper = [zeros(len_time_series, num_state) for i in 1:num_dim]
    ALPHA_nume_lower = [zeros(len_time_series, num_state) for i in 1:num_dim]
    ## for each dimension d, ALPHA_nume[2:end] is also the forecast pseudo-residual!
    C = zeros(len_time_series, 1)  # rescaling factors

    GC.safepoint()

    # prepare numerator and denominator components for time 1
    ALPHA_denom[1, :] = transpose(π_list)   # only

    ## ordinary numerator pieces at time 1
    @threads for d in 1:num_dim
        data_emiss_cdf_upper_d = data_emiss_cdf_list_separate[d]
        data_emiss_cdf_lower_d = data_emiss_cdf_lower_list_separate[d]

        ALPHA_nume_upper[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_upper_d[1, :])
        ALPHA_nume_lower[d][1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_cdf_lower_d[1, :])

        GC.safepoint()
    end

    ## init alpha at time 1
    ALPHA[1, :] = transpose(ALPHA_denom[1, :]) * Diagonal(data_emiss_prob_list_total[1, :])

    # scaling
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1.0 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]

    for v = 2:len_time_series

        ALPHA_denom[v, :] = transpose(ALPHA[v - 1, :]) * Pt_list[v - 1] # only

        ## forecast pseudo-residuals at time v use ALPHA_denom[v, :]
        @threads for d in 1:num_dim
            data_emiss_cdf_upper_d = data_emiss_cdf_list_separate[d]
            data_emiss_cdf_lower_d = data_emiss_cdf_lower_list_separate[d]

            ALPHA_nume_upper[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_upper_d[v, :])
            ALPHA_nume_lower[d][v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_cdf_lower_d[v, :])

            GC.safepoint()
        end

        ALPHA[v, :] = transpose(ALPHA_denom[v, :]) * Diagonal(data_emiss_prob_list_total[v, :])

        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1.0 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]

    end # v

    GC.safepoint()

    ## compute beta()
    BETA = zeros(len_time_series, num_state)

    # init beta()
    BETA[len_time_series, :] .= C[len_time_series]

    for v = (len_time_series - 1):(-1):1
        BETA[v, :] = Pt_list[v] * Diagonal(data_emiss_prob_list_total[v + 1, :]) * BETA[v + 1, :]
        BETA[v, :] = BETA[v, :] .* C[v]
    end

    GC.safepoint()

    ## group denominator
    denominator = sum(ALPHA_denom .* BETA, dims = 2)

    ordinary_upper = zeros(len_time_series, num_dim)
    ordinary_lower = zeros(len_time_series, num_dim)

    @threads for d in 1:num_dim
        ordinary_upper[:, d] = sum(ALPHA_nume_upper[d] .* BETA, dims = 2) ./ denominator
        ordinary_lower[:, d] = sum(ALPHA_nume_lower[d] .* BETA, dims = 2) ./ denominator
    end

    ## forecast residuals obtained by summing predictive-state-weighted CDF contributions
    forecast_upper = zeros(len_time_series - 1, num_dim)
    forecast_lower = zeros(len_time_series - 1, num_dim)

    @threads for d = 1:num_dim
        forecast_upper[:, d] = sum(ALPHA_nume_upper[d][2:end, :], dims = 2)
        forecast_lower[:, d] = sum(ALPHA_nume_lower[d][2:end, :], dims = 2)
    end

    ordinary_upper .= clamp.(ordinary_upper, 0.0, 1.0)
    ordinary_lower .= clamp.(ordinary_lower, 0.0, 1.0)
    forecast_upper .= clamp.(forecast_upper, 0.0, 1.0)
    forecast_lower .= clamp.(forecast_lower, 0.0, 1.0)

    ordinary = randomized_pit(ordinary_upper, ordinary_lower)
    forecast = randomized_pit(forecast_upper, forecast_lower)

    if uniform == 0
        ordinary = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), ordinary)
        forecast = HMMToolkit.quantile.(HMMToolkit.NormalExpert(0, 1), forecast)
    end

    if keep_last == 0
        ordinary = ordinary[1:(end - 1), :]
        forecast = forecast[1:(end - 1), :]
    end

    return (ordinary_residuals = ordinary, forecast_residuals = forecast)

end



"""
    CTHMM_batch_pseudo_residuals(uniform, df, response_list, Q_mat, π_list, state_list;
        keep_last = 1, group_by_col = nothing)

Computes both the uniform or normal ordinary and forecast pseudo residuals of (multiple) time series in batches given a fitted CTHMM.

# Arguments
- `uniform`: 1 for uniform ordinary pseudo residuals, normal otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Fitted transition rate matrix.
- `π_list`: Fitted initial state probabilities.
- `state_list`: Fitted state dependent distributions.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.

# Return Values
- `ordinary_list`: A list of g ordinary pseudo-residual array for g time series.
- `forecast_list`: A list of g forecast pseudo-residual array for g time series.

`ZI_list` is accepted for compatibility; zero inflation is inferred from `state_list`.
"""

function CTHMM_batch_pseudo_residuals(uniform, df, response_list, Q_mat, π_list, state_list; ZI_list = nothing, keep_last = 1, group_by_col = nothing)
    
    # batch compute both the ordinary and forecast pseudo-residuals
    # uniform = 1 if uniform; uniform = 0 if normal
    
    # keep_last = 1 if the last forecast is to be kept; 
    # keep_last = 0 if the last forecast is to be removed - in the case of missing observation (e.g. acceleration, angle change)

    ## precomputation
    obs_seq_emiss_list_total = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list; group_by_col = group_by_col)
    obs_seq_cdf_list_separate = CTHMM_precompute_batch_data_emission_cdf_separate(df, response_list, state_list; group_by_col = group_by_col)
    obs_seq_cdf_lower_list_separate = CTHMM_precompute_batch_data_emission_cdf_lower_separate(df, response_list, state_list; group_by_col = group_by_col)

    num_state = size(Q_mat, 1)

    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end

    num_time_series = size(group_df, 1)

    ordinary_list = Array{Array{Float64}}(undef, num_time_series)
    forecast_list = Array{Array{Float64}}(undef, num_time_series)
    
    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = group_df[g]

        data_emiss_prob_list_total = obs_seq_emiss_list_total[g][1]
        data_emiss_cdf_list_separate = obs_seq_cdf_list_separate[g]
        data_emiss_cdf_lower_list_separate = obs_seq_cdf_lower_list_separate[g]
        ordinary_list[g], forecast_list[g] = CTHMM_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, data_emiss_cdf_lower_list_separate, Q_mat, π_list; keep_last = keep_last)

        GC.safepoint()
        
    end # g

    GC.safepoint()

    return (ordinary_list = ordinary_list, forecast_list = forecast_list)
    
end




"""
    CTHMM_anomaly_indices(forecast_list)

Computes the anomaly indices given a list of normal forecast pseudo-residuals.

# Arguments
- `forecast_list`: A list of g normal forecast pseudo-residual array for g time series; returned by `CTHMM_batch_pseudo_residuals` function (2nd arg).

# Return Values
A g-by-(num_dim) array storing the anomaly indices of the g time series, for each of the (num_dim) response dimensions.
"""

function CTHMM_anomaly_indices(forecast_list)
    
    num_time_series = length(forecast_list)
    num_dim = size(forecast_list[1])[2]

    anomaly_indices = Array{Float64}(undef, num_time_series, num_dim)

    @threads for j in 1:num_time_series

        forecast = forecast_list[j]
        forecast[isnan.(forecast)] .= -10
        forecast[forecast .== Inf] .= 10

        anomaly_indices[j, :] = [mean(abs.(forecast[:, d]) .>= 3) for d in 1:num_dim]

    end

    return anomaly_indices
    
end



"""
    CTHMM_batch_anomaly_indices(df, response_list, Q_mat, π_list, state_list;
        keep_last = 1, group_by_col = nothing)

Computes the anomaly indices of (multiple) time series in batches given a fitted CTHMM.

# Arguments
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Fitted transition rate matrix.
- `π_list`: Fitted initial state probabilities.
- `state_list`: Fitted state dependent distributions.

# Optional Arguments
- `keep_list`: 1 (default) if last observation is kept, removed otherwise.

# Return Values
- A g-by-(num_dim) array storing the anomaly indices of the g time series, for each of the (num_dim) response dimensions.

`ZI_list` is accepted for compatibility; zero inflation is inferred from `state_list`.
"""

function CTHMM_batch_anomaly_indices(df, response_list, Q_mat, π_list, state_list; ZI_list = nothing, keep_last = 1, group_by_col = nothing)

    # uniform = 0 for normal pseudo-residuals
    ordinary_list, forecast_list = CTHMM_batch_pseudo_residuals(0, df, response_list, Q_mat, π_list, state_list; 
                                                                    keep_last = keep_last, group_by_col = group_by_col)
    
    anomaly_indices = CTHMM_anomaly_indices(forecast_list)

    return anomaly_indices

end


# Compatibility with the previous response_list / ZI_list low-level interface.
function CTHMM_ordinary_pseudo_residuals(uniform, seq_df, response_list::AbstractVector, data_emiss_prob_list_total, data_emiss_cdf_list_separate, Q_mat, π_list; keep_last = 1, ZI_list = nothing)
    cdf_lower = _legacy_residual_cdf_lower(seq_df, response_list, data_emiss_cdf_list_separate, ZI_list)
    return CTHMM_ordinary_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, cdf_lower, Q_mat, π_list; keep_last = keep_last)
end

function CTHMM_forecast_pseudo_residuals(uniform, seq_df, response_list::AbstractVector, data_emiss_prob_list_total, data_emiss_cdf_list_separate, Q_mat, π_list; keep_last = 1, ZI_list = nothing)
    cdf_lower = _legacy_residual_cdf_lower(seq_df, response_list, data_emiss_cdf_list_separate, ZI_list)
    return CTHMM_forecast_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, cdf_lower, Q_mat, π_list; keep_last = keep_last)
end

function CTHMM_pseudo_residuals(uniform, seq_df, response_list::AbstractVector, data_emiss_prob_list_total, data_emiss_cdf_list_separate, Q_mat, π_list; keep_last = 1, ZI_list = nothing)
    cdf_lower = _legacy_residual_cdf_lower(seq_df, response_list, data_emiss_cdf_list_separate, ZI_list)
    return CTHMM_pseudo_residuals(uniform, seq_df, data_emiss_prob_list_total, data_emiss_cdf_list_separate, cdf_lower, Q_mat, π_list; keep_last = keep_last)
end
