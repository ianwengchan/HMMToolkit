function CTHMM_learn_para_EM(ori_method, max_iter, ori_is_outer_soft, Q_mat_init, state_init_prob_list_init, state_list_init, df)

    # ## start timer()
    # tStart = tic()

    ## precomputation before iteration
    # only once in the whole EM:
    distinct_time_list = CTHMM_precompute_distinct_time_list(df.time_interval)
    num_state = size(Q_mat_init, 1)
    num_dim = size(state_list_init, 1)
    num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series
    # have to redo for updated Q and distribution parameters:
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat_init)  # with initial guess Q0
    obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, state_list_init)  # with initial parameter guess
    
    ## iteration
    pre_all_subject_prob = -Inf
    model_iter_count = 0
    
    ## init learn performance; compute initial likelihood??
    CTHMM_learn_init_performance[ori_method, is_outer_soft, max_iter]

    Q_mat = Q_mat_init
    state_init_prob_list = state_init_prob_list_init

    while (model_iter_count < max_iter)
        
        # tStartIter = tic
        
        ## add counter
        model_iter_count = model_iter_count + 1
            
        ## E-step
        ## batch soft decoding (option = 1), saving Svi to df
        cur_all_subject_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(1, df, response_list, Q_mat, state_init_prob_list, state_list)
        
        ## M-step
        ## part 1: learning Q_mat for every distinct time
        Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
        Q_mat_new = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

        ## part 2: learning state dependent distribution parameters
        for d in 1:num_dim
            for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)])
            end
        end

        



        

        tEndTemp = toc(tStartTemp)
        str = sprintf("\nCompute all inner counts: #d minutes & #f seconds\n", floor(tEndTemp/60),rem(tEndTemp,60))
        CTHMM_print_log[str];    
        learn_performance.time_inner_list[model_iter_count] = tEndTemp;    
        ## ===========================
        tEndIter = toc(tStartIter)
        str = sprintf("Iter #d: #d minutes & #f seconds\n", model_iter_count, floor(tEndIter/60),rem(tEndIter,60))
        CTHMM_print_log[str]
        learn_performance.time_list[model_iter_count] = tEndIter;        
        ## ===========================
        
        ## compute current learning performance
        CTHMM_learn_record_performance[]
        
        ## Update Q_mat: qij = Nij/Ti
        pre_Q_mat = Q_mat
        CTHMM_learn_update_Q_mat[]
        Q_mat
        
        ## Store main variables in top_out_folder
        CTHMM_learn_store_main_variables[top_out_folder]
        
        ## output the time for one iteration   
        tEnd = toc(tStart)
        str = sprintf("Total elapse time: #d minutes & #f seconds\n", floor(tEnd/60),rem(tEnd,60))
        CTHMM_print_log[str]
        
        ## check if reached a fixed point
        [is_termindate] = CTHMM_learn_decide_termination[cur_all_subject_prob, pre_all_subject_prob];    
        if (is_termindate .== 1)
            str = sprintf("#s/num_iter.txt", out_dir)
            fp = fopen(str, "wt")
            fprintf(fp, "#d\n", model_iter_count)
            fclose(fp)
            if (cur_all_subject_prob .< pre_all_subject_prob)
                Q_mat = pre_Q_mat
            end
            break()
        end
        
        ## store current all subject prob
        pre_all_subject_prob = cur_all_subject_prob
        
    end # ite
    
    ## record performance
    CTHMM_learn_stop_record_performance[]; 
    
    tEnd = toc(tStart)
    str = sprintf("Total elapse time: #d minutes & #f seconds\n", floor(tEnd/60),rem(tEnd,60))
    CTHMM_print_log[str]
    learn_performance.total_time = tEnd
    sprintf("Q matrix below")
    Q_mat
    
end