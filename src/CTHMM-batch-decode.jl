## NO COVARIATES

"""
    CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

Batch decodes `Etij` for (multiple) time series using soft decoding (forward-backward) or hard decoding (viterbi), 
given dataframe `df`,
list of responses `response_list`,
transition rate matrix  `Q_mat`,
initial state probabilities `π_list`,
and state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Transition rate matrix.
- `π_list`: Initial state probabilities.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- `Svi_full`: Smoothers `Svi`, expected state probabilities, for all time series.
- `cur_all_subject_prob`: Sum of loglikelihood of all time series.
- `Etij_all`: Soft count tables `Etij`, expected transition probability matrices, for each distinct time interval t.
"""

function CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

    ## precomputation
    time_interval_list = df.time_interval
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list; group_by_col = group_by_col)

    num_distinct_time = size(distinct_time_list, 1) # from all time series
    num_state = size(Q_mat, 1)

    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end

    num_time_series = size(group_df, 1)

    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)
    Etij_list = Array{Array{Float64, 3}}(undef, num_time_series)
    Svi_list = Array{Array{Float64}}(undef, num_time_series)

    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = group_df[g]
        len_time_series = nrow(seq_df)
        seq_time_interval_list = collect(skipmissing(seq_df.time_interval))

        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (soft_decode == 1) # soft decoding
            data_emiss_prob_list = obs_seq_emiss_list[g][1]
            Svi, Evij, subject_log_prob = CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, π_list)
        else
            log_data_emiss_prob_list = obs_seq_emiss_list[g][2]
            state_seq, subject_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)
            Svi = zeros(len_time_series, num_state)
            for i in 1:num_state
                Svi[:, i] = vec(map(x -> x == i ? 1 : 0, state_seq))
            end
        end

        GC.safepoint()

        Svi_list[g] = Svi
        cur_all_subject_prob_list[g] = subject_log_prob  # log prob from either soft or hard decoding
    
        ## soft count table C(delta, k, l), denoted Etij, to be accumulated from Evij
        Etij = zeros(num_distinct_time, num_state, num_state)

        ## group Evij to be Etij
        for v = 1:(len_time_series-1)

            ## find T in distinct_time_list and corresponding t_idx (only one)
            T = seq_time_interval_list[v]
            t_idx = findfirst(x -> x .== T, distinct_time_list)

            if (soft_decode == 1)
                Etij[t_idx, :, :] = Etij[t_idx, :, :] + Evij[v, :, :]
            else
                i = state_seq[v]
                j = state_seq[v+1]
                Etij[t_idx, i, j] = Etij[t_idx, i, j] + 1
            end

            GC.safepoint()
        end # v

        Etij_list[g] = Etij

        GC.safepoint()
        
    end # g

    GC.safepoint()

    cur_all_subject_prob = sum(cur_all_subject_prob_list)
    Etij_all = sum(Etij_list)

    Svi_full = reduce(vcat, Svi_list)

    return (Svi_full = Svi_full, cur_all_subject_prob = cur_all_subject_prob, Etij_all = Etij_all)
    
end



"""
    CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

Batch decodes `Etij` for (multiple) time series using soft decoding (forward-backward) or hard decoding (viterbi), 
and append smoothers `Svi`, expected state probabilities, to the dataframe `df`,
given dataframe `df`,
list of responses `response_list`,
transition rate matrix  `Q_mat`,
initial state probabilities `π_list`,
and state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Transition rate matrix.
- `π_list`: Initial state probabilities.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- `cur_all_subject_prob`: Sum of loglikelihood of all time series.
- `Etij_all`: Soft count tables `Etij`, expected transition probability matrices, for each distinct time interval t.
"""

function CTHMM_batch_decode_Etij_and_append_Svi_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

    Svi_full, cur_all_subject_prob, Etij_all = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)

    num_state = size(Q_mat, 1)
    for i in 1:num_state
        df[:, string("Sv", i)] = Svi_full[:, i]
    end

    GC.safepoint()

    return (cur_all_subject_prob = cur_all_subject_prob, Etij_all = Etij_all)

end



"""
    CTHMM_batch_decode_for_subjects_list(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

Batch decodes and returns a list of loglikelihood for (multiple) time series (separately) using soft decoding (forward-backward) or hard decoding (viterbi), 
given dataframe `df`,
list of responses `response_list`,
transition rate matrix  `Q_mat`,
initial state probabilities `π_list`,
and state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Transition rate matrix.
- `π_list`: Initial state probabilities.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- A list of loglikelihood for each time series.
"""

function CTHMM_batch_decode_for_subjects_list(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

    # returning the loglikelihood for EACH time series in a list!! NOT CUMULATIVE!!
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list; group_by_col = group_by_col)

    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end

    num_time_series = size(group_df, 1)

    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)

    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = group_df[g]

        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (soft_decode == 1) # soft decoding
            data_emiss_prob_list = obs_seq_emiss_list[g][1]
            subject_log_prob = CTHMM_likelihood_forward(seq_df, data_emiss_prob_list, Q_mat, π_list)
            # DO NOT UPDATE Svi
        else
            log_data_emiss_prob_list = obs_seq_emiss_list[g][2]
            state_seq, subject_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, π_list)
            # DO NOT UPDATE Svi
        end

        cur_all_subject_prob_list[g] = subject_log_prob # log prob from either soft or hard decoding

        GC.safepoint()
        
    end # g

    GC.safepoint()

    return cur_all_subject_prob_list
    
end



"""
    CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

Batch decodes and returns the total loglikelihood for (multiple) time series using soft decoding (forward-backward) or hard decoding (viterbi), 
given dataframe `df`,
list of responses `response_list`,
transition rate matrix  `Q_mat`,
initial state probabilities `π_list`,
and state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `soft_decode`: 1 for soft decoding (forward-backward), hard decoding (viterbi) otherwise.
- `df`: Dataframe of (multiple) time series.
- `response_list`: List of responses to consider.
- `Q_mat`: Transition rate matrix.
- `π_list`: Initial state probabilities.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- Total loglikelihood for all time series.
"""

function CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = nothing)

    # returning the total loglikelihood for all time series!! CUMULATIVE!!
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    cur_all_subject_prob = 0.0
    
    cur_all_subject_prob_list = CTHMM_batch_decode_for_subjects_list(soft_decode, df, response_list, Q_mat, π_list, state_list; group_by_col = group_by_col)
    
    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    return cur_all_subject_prob
    
end