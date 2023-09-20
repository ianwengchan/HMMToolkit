function CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, π_list; ϵ = 1e-08)

    # decoding for only one time-series
    # even if i do decoding for all time-series (having the same covariates)
    # i only have to compute the distinct_time_list and distinct_time_Pt_list for once

    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # already feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    ## precompute data emission prob
    ## obs_seq_emiss_list[g][1] # data_emiss_prob_list
    ## obs_seq_emiss_list[g][2] # log_data_emiss_prob_list

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    C = zeros(len_time_series, 1) # rescaling factor

    ## precomputing of Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
    end

    ## init alpha for time 1
    ALPHA[1, :] = π_list .* data_emiss_prob_list[1, :]
    C[1] = 1.0 / (sum(ALPHA[1, :]) + ϵ)
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    for v = 2:len_time_series

        for s = 1:num_state # current state
            prob = 0.0
            prob = prob + sum(ALPHA[v-1, :] .* Pt_list[v-1][:, s]) * data_emiss_prob_list[v, s]
            ALPHA[v, s] = prob   
        end # s

        # scaling
        C[v] = 1.0 / (sum(ALPHA[v, :]) + ϵ)
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
    end # v

    
    ## compute beta()
    BETA = zeros(len_time_series, num_state)

    # init beta()
    BETA[len_time_series, :] .= C[len_time_series]
    
    for v = (len_time_series-1):(-1):1

        for s = 1:num_state # current state
            prob = 0.0
            prob = prob + sum(Pt_list[v][s, :] .* data_emiss_prob_list[v+1, :] .* BETA[v+1, :])
            BETA[v, s] = prob
        end # s

        BETA[v, :] = BETA[v, :] .* C[v]
    end # v

    ## compute Svi, smoother
    Svi = ALPHA .* BETA ./ C

    ## compute Evij, two-slice marginal
    Evij = zeros((len_time_series-1), num_state, num_state)

    @threads for v = 1:(len_time_series-1)

        for i = 1:num_state
            Evij[v, i, :] = ALPHA[v, i] * Pt_list[v][i, :] .* data_emiss_prob_list[v+1, :] .* BETA[v+1, :]
        end
        sum_Evij = sum(Evij[v, :, :])
        Evij[v, :, :] = Evij[v, :, :] ./ sum_Evij
    
    end
    
    ## check rabinar's paper
    log_prob = 0
    log_prob = log_prob - sum(log.(C))

    return Svi, Evij, log_prob
    
end


function CTHMM_likelihood_forward(seq_df, data_emiss_prob_list, Q_mat, π_list; ϵ = 1e-08)

    # computing the likelihood via soft decoding only needs the forward step

    num_state = size(Q_mat, 1)
    len_time_series = nrow(seq_df)
    time_interval_list = collect(skipmissing(seq_df.time_interval))   # already feed-in the time intervals

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    ## precompute data emission prob
    ## obs_seq_emiss_list[g][1] # data_emiss_prob_list
    ## obs_seq_emiss_list[g][2] # log_data_emiss_prob_list

    ## compute alpha()
    ALPHA = zeros(len_time_series, num_state)
    C = zeros(len_time_series, 1) # rescaling factor

    ## precomputing of Pt for every timepoint
    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
    end

    ## init alpha for time 1
    ALPHA[1, :] = π_list .* data_emiss_prob_list[1, :]
    C[1] = 1.0 / (sum(ALPHA[1, :]) + ϵ)
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    for v = 2:len_time_series

        for s = 1:num_state # current state
            prob = 0.0
            prob = prob + sum(ALPHA[v-1, :] .* Pt_list[v-1][:, s]) * data_emiss_prob_list[v, s]
            ALPHA[v, s] = prob   
        end # s

        # scaling
        C[v] = 1.0 / (sum(ALPHA[v, :]) + ϵ)
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
    end # v
    
    ## check rabinar's paper
    log_prob = 0
    log_prob = log_prob - sum(log.(C))

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

    Pt_list = Array{Matrix{Float64}}(undef, (len_time_series-1))
    @threads for v = 1:(len_time_series-1)
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
    end

    v = 1
    log_prob = log((π_list .* data_emiss_prob_list[1, :])[true_state[1]])
    for v = 2:len_time_series
        i = true_state[v-1]
        j = true_state[v]
        log_prob = log_prob + log(Pt_list[v-1][i, j] * data_emiss_prob_list[v, j])
    end

    return log_prob
    
end