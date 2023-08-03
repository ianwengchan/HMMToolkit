function [cur_all_subject_prob] = CTHMM_learn_batch_outer_decoding_Etij_for_subjects(is_outer_soft, train_idx_list)

    disp("CTHMM_batch_outer_decoding_Etij_for_subjects")
    tStartTemp = tic()

    cur_all_subject_prob = 0.0
    num_distinct_time = size(distinct_time_list, 1) # from all time series
    num_state = size(state_list, 1)

    ## soft count table C(delta, k, l), to be accumulated from Evij
    Etij = zeros(num_distinct_time, num_state, num_state)

    num_time_series = size(train_idx_list, 1)
    
    # model_iter_count
    
    for g = 1:num_time_series
                
        if (g % 10 .== 0)
            str = sprintf("#d...", g)
            fprintf(str)
        end

        ## get the subject index
        len_time_series = obs_seq_list[g].num_visit
        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        if (is_outer_soft == 1)
            [Evij, subject_log_prob, Pt_list] = CTHMM_decode_outer_forward_backward(obs_seq_list[subject_idx])
        else()
            [outer_state_seq, outer_dur_seq, subject_log_prob, Pt_list] = CTHMM_decode_outer_viterbi(obs_seq_list[subject_idx])
        end
                      
        # if (subject_log_prob .== inf)
        #     str = sprintf("subject_log_prob = inf for subject = #d\n", g)
        #     CTHMM_print_log[str]
        # end
        
        ## accum all prob from all time series
        cur_all_subject_prob = cur_all_subject_prob + subject_log_prob # soft prob
    
        ## group Evij to be Etij
        # num_distinct_time = size(distinct_time_list, 1)
        
        for v = 1:len_time_series

            ## find T in distinct_time_list and corresponding t_idx (only one)
            T = time_interval_list[v]
            t_idx = find(distinct_time_list .= T)

            if (is_out_soft == 1)
                Etij[t_idx, :, :] = Etij[t_idx, :, :] + Evij[v, :, :]
            else()
                i = outer_state_seq[v]
                j = outer_state_seq[v+1]
                Etij[t_idx, i, j] = Etij[t_idx, i, j] + 1
            end
        end # v
        
    end # g [subject index]
    
    learn_performance.time_outer_list[model_iter_count] = tEndTemp
    
end