function CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, Q_mat, state_init_prob_list)

    cur_all_subject_prob = 0.0

    ## precomputation
    time_interval_list = df.time_interval
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    # distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df)

    num_distinct_time = size(distinct_time_list, 1) # from all time series
    num_state = size(Q_mat, 1)

    ## soft count table C(delta, k, l), denoted Etij, to be accumulated from Evij
    Etij = zeros(num_distinct_time, num_state, num_state)

    group_df = groupby(df, :ID)
    num_time_series = size(group_df, 1)

    for g = 1:num_time_series
                
        if (g % 100 .== 0)
            println(g)
        end

        seq_df = group_df[g]
        len_time_series = nrow(seq_df)
        seq_time_interval_list = collect(skipmissing(seq_df.time_interval))

        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (soft_decode == 1) # soft decoding
            data_emiss_prob_list = obs_seq_emiss_list[g][1]
            Svi, Evij, subject_log_prob = CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, state_init_prob_list)
            for i in 1:num_state
                seq_df[:, string("Sv", i)] = Svi[:, i]
            end
        else()
            log_data_emiss_prob_list = obs_seq_emiss_list[g][2]
            state_seq, subject_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, state_init_prob_list)
        end


        ## accum all prob from all time series
        cur_all_subject_prob = cur_all_subject_prob + subject_log_prob # log prob from either soft or hard decoding
    
        ## group Evij to be Etij        
        for v = 1:(len_time_series-1)

            ## find T in distinct_time_list and corresponding t_idx (only one)
            T = seq_time_interval_list[v]
            t_idx = findfirst(x -> x .== T, distinct_time_list)

            if (soft_decode == 1)
                Etij[t_idx, :, :] = Etij[t_idx, :, :] + Evij[v, :, :]
            else()
                i = state_seq[v]
                j = state_seq[v+1]
                Etij[t_idx, i, j] = Etij[t_idx, i, j] + 1
            end
        end # v
        
    end # g
    
    # learn_performance.time_outer_list[model_iter_count] = tEndTemp

    return cur_all_subject_prob, Etij
    
end