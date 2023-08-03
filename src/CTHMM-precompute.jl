function CTHMM_precompute_batch_data_emission_prob(train_idx_list)

    disp("In CTHMM_precompute_batch_data_emission_prob()...")
    
    global obs_seq_list
    global state_list
    
    ## data emission probability
    
    num_time_series = size(train_idx_list, 1)
    num_state = size(state_list, 1)
    
    for g = 1:num_time_series   # number of time series to consider, e.g. trips
          
        if (g % 10 .== 0)
            str = sprintf("#d...", g)
            fprintf(str)
        end
    
        ## get the subject index
        # subject_idx = train_idx_list[g]
        len_time_series = obs_seq_list[g].num_visit

        # subject data        
        obs_seq_list[g].data_emiss_prob_list = zeros(len_time_series, num_state)
        obs_seq_list[g].log_data_emiss_prob_list = zeros(len_time_series, num_state)
    
        ## compute forward backward algorithm, remember Et[i,j] parameter: prob at the end-states at time t
        for v = 1:num_visit    
            data = obs_seq_list[subject_idx].visit_data_list[v, :]
    
            for s = 1:num_state
                emiss_prob = mvnpdf[data, state_list[s].mu, state_list[s].var]  # assume normal at this point; change probability distributions later
                #data_emiss_prob_mat[v, s] = func_state_emiss[s, data]
                obs_seq_list[subject_idx].data_emiss_prob_list[v,s] = emiss_prob
                obs_seq_list[subject_idx].log_data_emiss_prob_list[v,s] = log(emiss_prob)
            end
        end
            
    end

end



function CTHMM_learn_onetime_precomputation(train_idx_list)

    global is_use_distinct_time_grouping
    global distinct_time_list
    global learn_performance
    
    ## precomputation of state emission prob of observations
    tStartTemp= tic()

    ## find out distinct time from the dataset
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)


    if (is_use_distinct_time_grouping .== 1)
        [distinct_time_list] = CTHMM_precompute_distinct_time_intv[train_idx_list]
    end
    ## comute all data emission probability
    CTHMM_precompute_batch_data_emission_prob[train_idx_list]
    tEndTemp = toc(tStartTemp)
    
    ## time in precomputation
    str = sprintf("\nPrecomputation of data emission-> time: #d minutes & #f seconds\n", floor(tEndTemp/60),rem(tEndTemp,60))
    CTHMM_print_log[str]
    
    learn_performance.time_precomp_data = tEndTemp;  
end