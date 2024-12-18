"""
    CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)

Computes viterbi decoding for a single time series, 
given time series `seq_df`,
its logged data emission probability list `log_data_emiss_prob_list`,
transition rate matrix  `Q_mat`,
and initial state probabilities `π_list`.

# Arguments
- `seq_df`: Dataframe of a single time series.
- `log_data_emiss_prob_list`: Logged data emission probability list of the time series; returned by `CTHMM_precompute_batch_data_emission_prob` function (2nd arg for the time series).
- `Q_mat`: Transition rate matrix.
- `π_list`: Initial state probabilities.

# Return Values
- `best_state_seq`: A sequence of most likely (hidden) states for the time series.
- `best_log_prob`: Loglikelihood of the time series corresponding to `best_state_seq`.
"""

function CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)

    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval)) # feed-in the time intervals
    
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    best_state_seq = zeros(Int8, len_time_series, 1)    # Int8 for indexing

    ## precompute data emission prob
    ## obs_seq_emiss_list[g][1] # data_emiss_prob_list
    ## obs_seq_emiss_list[g][2] # log_data_emiss_prob_list

    ## create two matrices T1 and T2 to store decoding results
    T1 = -Inf * ones(num_state, len_time_series)  # tracking best prob so far
    T2 = zeros(Int8, num_state, len_time_series)  # backtracking pointer, Int8 for indexing

    GC.safepoint()

    ## precomputing of Pt for every visit
    log_Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        log_Pt_list[v] = log.(distinct_time_Pt_list[t_idx])
        GC.safepoint()
    end

    GC.safepoint()

    ## init T1 and T2 at first timepoint
    max_log_prob = -Inf

    log_emiss_prob = log_data_emiss_prob_list[1, :]
    log_prob = log.(π_list) + log_emiss_prob

    T1[:, 1] = log_prob
    T2[:, 1] .= 0    # at time 0 there is no previous state

    if (maximum(log_prob) >= max_log_prob)
        max_log_prob = maximum(log_prob)
    end

    GC.safepoint()
    
    ## for all other timepoints
    for v = 2:len_time_series  
        
        for s = 1:num_state # current state
            
            ## different state has different best previous state
            log_emiss_prob = log_data_emiss_prob_list[v, s]
            log_prob = T1[:, v-1] + log_Pt_list[v-1][:, s] .+ log_emiss_prob

            ## find best previous state
            best_log_prob, best_k = findmax(log_prob)
            
            T1[s, v] = best_log_prob
            T2[s, v] = best_k
        end # s
    end # v

    GC.safepoint()
    
    
    ## start backtracking of state sequence
    ## find best last state
    best_log_prob, best_last_s = findmax(T1[:, len_time_series])
    best_state_seq[len_time_series] = best_last_s
    
    ## start backtracking from the last state
    for v = (len_time_series-1):(-1):1
        best_state_seq[v] = T2[best_last_s, v+1]
        best_last_s = best_state_seq[v]
    end

    GC.safepoint()

    return (best_state_seq = best_state_seq, best_log_prob = best_log_prob)
    
end