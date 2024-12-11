function CTHMM_batch_decode_for_subjects_list_instance(df_list, response_list, Q_mat, state_list)

    # returning the loglikelihood for EACH subject in a list!! NOT CUMULATIVE!!
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    for (idx, df) in enumerate(df_list)
        df.group .= idx
    end
    df_tmp = vcat(df_list...)
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df_tmp, response_list, state_list; group_by_col = [:group])

    num_time_series = size(df_list, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)
    duration_list = Array{Float64}(undef, num_time_series)
    count_list = Array{Float64}(undef, num_time_series)

    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = df_list[g]

        data_emiss_prob_list = obs_seq_emiss_list[g][1]
        subject_log_prob = CTHMM_likelihood_instance(seq_df, data_emiss_prob_list, Q_mat, seq_df.best_state)

        cur_all_subject_prob_list[g] = subject_log_prob
        duration_list[g] = maximum(seq_df.time_since) - minimum(seq_df.time_since)
        count_list[g] = nrow(seq_df)

        GC.safepoint()
        
    end # g

    GC.safepoint()

    return cur_all_subject_prob_list, duration_list, count_list
    
end



function DTHMM_batch_decode_for_subjects_list_instance(df_list, response_list, P_mat, state_list)

    # returning the loglikelihood for EACH subject in a list!! NOT CUMULATIVE!!
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    for (idx, df) in enumerate(df_list)
        df.group .= idx
    end
    df_tmp = vcat(df_list...)
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df_tmp, response_list, state_list; group_by_col = [:group])

    num_time_series = size(df_list, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_time_series)
    duration_list = Array{Float64}(undef, num_time_series)
    count_list = Array{Float64}(undef, num_time_series)

    GC.safepoint()

    @threads for g = 1:num_time_series

        seq_df = df_list[g]

        data_emiss_prob_list = obs_seq_emiss_list[g][1]
        subject_log_prob = DTHMM_likelihood_instance(seq_df, data_emiss_prob_list, P_mat, seq_df.best_state)

        cur_all_subject_prob_list[g] = subject_log_prob
        duration_list[g] = maximum(seq_df.time_since) - minimum(seq_df.time_since)
        count_list[g] = nrow(seq_df)

        GC.safepoint()
        
    end # g

    GC.safepoint()

    return cur_all_subject_prob_list, duration_list, count_list
    
end