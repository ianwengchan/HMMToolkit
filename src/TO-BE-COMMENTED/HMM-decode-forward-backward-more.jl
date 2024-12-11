function CTHMM_likelihood_instance(seq_df, data_emiss_prob_list, Q_mat, true_state)

    # computing the likelihood via soft decoding, requires only the forward step

    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # already feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    GC.safepoint()

    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end

    GC.safepoint()

    v = 1
    log_prob = log(data_emiss_prob_list[1, true_state[1]])  # without initial state probability

    if (len_time_series > 1)
        for v = 2:len_time_series
            i = true_state[v-1]
            j = true_state[v]
            log_prob = log_prob + log(Pt_list[v-1][i, j] * data_emiss_prob_list[v, j])
        end
    end
    
    GC.safepoint()

    return log_prob
    
end



function CTHMM_likelihood_true(seq_df, data_emiss_prob_list, Q_mat, π_list, true_state)

    # computing the likelihood via soft decoding only needs the forward step

    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # already feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    ## precompute data emission prob
    ## obs_seq_emiss_list[g][1] # data_emiss_prob_list
    ## obs_seq_emiss_list[g][2] # log_data_emiss_prob_list

    GC.safepoint()

    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
        GC.safepoint()
    end

    GC.safepoint()

    v = 1
    log_prob = log((π_list .* data_emiss_prob_list[1, :])[true_state[1]])
    for v = 2:len_time_series
        i = true_state[v-1]
        j = true_state[v]
        log_prob = log_prob + log(Pt_list[v-1][i, j] * data_emiss_prob_list[v, j])
    end
    
    GC.safepoint()

    return log_prob
    
end



function DTHMM_likelihood_instance(seq_df, data_emiss_prob_list, P_mat, true_state)

    # computing the likelihood via soft decoding only needs the forward step

    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    GC.safepoint()

    v = 1
    log_prob = log(data_emiss_prob_list[1, true_state[1]])  # without initial state probability

    if (len_time_series > 1)
        for v = 2:len_time_series
            i = true_state[v-1]
            j = true_state[v]
            log_prob = log_prob + log(P_mat[i, j] * data_emiss_prob_list[v, j])
        end
    end
    
    GC.safepoint()

    return log_prob
    
end


function DTHMM_likelihood_true(seq_df, data_emiss_prob_list, P_mat, π_list, true_state)

    # computing the likelihood via soft decoding only needs the forward step

    num_state = size(P_mat, 1)
    len_time_series = nrow(seq_df)

    GC.safepoint()

    v = 1
    log_prob = log((π_list .* data_emiss_prob_list[1, :])[true_state[1]])
    for v = 2:len_time_series
        i = true_state[v-1]
        j = true_state[v]
        log_prob = log_prob + log(P_mat[i, j] * data_emiss_prob_list[v, j])
    end
    
    GC.safepoint()

    return log_prob
    
end