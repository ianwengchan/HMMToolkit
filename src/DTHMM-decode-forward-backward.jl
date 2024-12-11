"""
    DTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, P_mat, π_list)

Computes forward-backward algorithm for a single time series, 
given time series `seq_df`,
its data emission probability list `data_emiss_prob_list`,
transition probability matrix  `P_mat`,
and initial state probabilities `π_list`.

# Arguments
- `seq_df`: Dataframe of a single time series.
- `data_emiss_prob_list`: Data emission probability list of the time series; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `P_mat`: Transition probability matrix.
- `π_list`: Initial state probabilities.

# Return Values
- `Svi`: Smoother , expected state probabilities at every time interval v.
- `Evij`: Two-slice marginal, expected transition probabilities at every time interval v.
- `log_prob`: Loglikelihood of the time series.
"""

function DTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, P_mat, π_list)

    # decoding for only one time-series

    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    ## init alpha for time 1
    ALPHA[1, :] = transpose(π_list) * Diagonal(data_emiss_prob_list[1, :])
    
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    for v = 2:len_time_series

        ALPHA[v, :] = transpose(ALPHA[v-1, : ]) * P_mat * Diagonal(data_emiss_prob_list[v, :])

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
        
        BETA[v, :] = P_mat * Diagonal(data_emiss_prob_list[v+1, :]) * BETA[v+1, :]

        # scaling
        BETA[v, :] = BETA[v, :] .* C[v]
    end # v

    GC.safepoint()

    ## compute Svi, smoother
    Svi = ALPHA .* BETA ./ C
    Svi[isnan.(Svi)] .= 0

    ## compute Evij, two-slice marginal
    Evij = zeros((len_time_series-1), num_state, num_state)

    GC.safepoint()

    @threads for v = 1:(len_time_series-1)

        for i = 1:num_state
            Evij[v, i, :] = ALPHA[v, i] * P_mat[i, :] .* data_emiss_prob_list[v+1, :] .* BETA[v+1, :]
        end
        sum_Evij = (sum(Evij[v, :, :]) == 0) ? 1 : sum(Evij[v, :, :])
        Evij[v, :, :] = Evij[v, :, :] ./ sum_Evij
        Evij[isnan.(Evij)] .= 0

        GC.safepoint()
    
    end

    GC.safepoint()
    
    ## check rabinar's paper
    log_prob = 0
    log_prob = log_prob - sum(log.(C))

    return (Svi = Svi, Evij = Evij, log_prob = log_prob)
    
end



"""
    DTHMM_likelihood_forward(seq_df, data_emiss_prob_list, P_mat, π_list)

Computes the likelihood for a single time series via soft-decoding using the forward step, 
given time series `seq_df`,
its data emission probability list `data_emiss_prob_list`,
transition probability matrix  `P_mat`,
and initial state probabilities `π_list`.

# Arguments
- `seq_df`: Dataframe of a single time series.
- `data_emiss_prob_list`: Data emission probability list of the time series; returned by `CTHMM_precompute_batch_data_emission_prob` function (1st arg for the time series).
- `P_mat`: Transition probability matrix.
- `π_list`: Initial state probabilities.

# Return Values
- Loglikelihood of the time series.
"""

function DTHMM_likelihood_forward(seq_df, data_emiss_prob_list, P_mat, π_list)

    # computing the likelihood via soft decoding only needs the forward step

    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    C = zeros(len_time_series, 1) # rescaling factor

    GC.safepoint()

    ## init alpha for time 1
    ALPHA[1, :] = transpose(π_list) * Diagonal(data_emiss_prob_list[1, :])
    C[1] = (sum(ALPHA[1, :]) == 0) ? 1 : (1.0 / sum(ALPHA[1, :]))
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    for v = 2:len_time_series
        
        ALPHA[v, :] = transpose(ALPHA[v-1, : ]) * P_mat * Diagonal(data_emiss_prob_list[v, :])
        
        # scaling
        C[v] = (sum(ALPHA[v, :]) == 0) ? 1 : (1.0 / sum(ALPHA[v, :]))
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
    end # v

    GC.safepoint()
    
    ## check rabinar's paper
    log_prob = 0
    log_prob = log_prob - sum(log.(C))

    return log_prob
    
end