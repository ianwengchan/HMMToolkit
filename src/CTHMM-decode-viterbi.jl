function [best_state_seq, dur_seq, best_log_prob, log_Pt_list] = CTHMM_decode_outer_viterbi(obs_seq)

    num_state = size(Q_mat, 1)
    len_time_series = obs_seq.num_visit
    time_interval_list = obs_seq.time_interval_list # already feed-in the time intervals
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    best_state_seq = zeros(len_time_series, 1)

    if (obs_seq.has_compute_data_emiss_prob != 1)
        for v = 1:num_visit
            data = obs_seq.visit_list[v].data
            for m = 1:num_state            
                emiss_prob = mvnpdf(data, state_list{m}.mu, state_list{m}.var)
                obs_seq.data_emiss_prob_list[v,m] = emiss_prob
                obs_seq.log_data_emiss_prob_list[v,m] = log(emiss_prob)
            end
        end
    end

    ## create two matrices T1 and T2 to store decoding results
    T1 = -Inf * ones(num_state, len_time_series)    # tracking best prob so far
    T2 = zeros(num_state, len_time_series)  # backtracking pointer

    ## precomputing of Pt for every visit
    log_Pt_list = Array{Any}(Undef, len_time_series)
    for v = 1:len_time_series
        ## compute transition matrix at each timepoint
        T = time_interval_list[v]
        t_idx = find(distinct_time_list .= T)
        log_Pt_list[v] = log(distinct_time_Pt_list[t_idx])
    end
    dur_seq = time_interval_list


    ## init T1 and T2 at first timepoint
    max_log_prob = -Inf

    for s = 1:num_state

        log_emiss_prob = obs_seq.log_data_emiss_prob_list[1, s]
        log_prob = log(state_init_prob_list[s]) + log_emiss_prob

        T1[s, 1] = log_prob
        T2[s, 1] = 0
        
        if (log_prob >= max_log_prob)
            max_log_prob = log_prob      
        end
    end
    
    ## for all other timepoints
    
    for v = 2:num_visit   
        
        for s = 1:num_state # current state
            
            ## different state has a different best previous state
            log_emiss_prob = obs_seq.log_data_emiss_prob_list[v, s]
            
            ## find best previous state
            # best_k = 0
            # best_log_prob = -Inf

            log_prob = T1[:, v-1] + log_Pt_list[v-1][:, s] .+ log_emiss_prob

            best_k = find_max(log_prob)[2]
            best_log_prob = maximum(log_prob)
            
            # for k = 1:num_state # previous state
               
            #    log_prob = T1[k, v-1] + log_Pt_list[v-1][k, s] + log_emiss_prob

            #     if (log_prob > best_log_prob)
            #         best_k = k
            #         best_log_prob = log_prob
            #     end
            # end # k
            
            T1[s, v] = best_log_prob
            T2[s, v] = best_k
        end # s
    end # v
    
    # dur_seq[end] = 0
    

    ## start backtracking of state sequence
    
    ## find best last state
    # best_last_s = 0
    # best_log_prob = -Inf

    best_last_s = find_max(T1[:, len_time_series])[2]
    best_log_prob = maximum(T1[:, len_time_series])
    best_state_seq[len_time_series] = best_last_s

    # for s = 1:num_state
    #     if (T1[s, len_time_series] >= best_log_prob)
    #         best_last_s = s
    #         best_log_prob = T1[s, len_time_series]
    #     end
    # end
    
    # if (best_last_s == 0)
    #    disp('best last s = 0');
    # end
    
    ## start backtracking from the last state
    for v = (num_visit-1):(-1):1
        best_state_seq[v] = T2[best_last_s, v+1]
        best_last_s = best_state_seq[v]
    end
    
end # func
    