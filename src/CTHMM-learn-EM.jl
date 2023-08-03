function CTHMM_learn_para_EM(ori_method, max_iter, ori_is_outer_soft, Q_mat_init, train_idx_list)

    ## start timer()
    tStart = tic()

    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat_init)  # with initial guess Q0

    ## one time precomputation before iteration

    CTHMM_learn_onetime_precomputation[train_idx_list]
    
    CTHMM_learn_init_common[]
    
    ## to decide whether to draw 2D fig
    num_state = size(state_list, 1)
    
    ## iteration
    pre_all_subject_prob = -Inf
    model_iter_count = 0
    
    ## init learn performance
    CTHMM_learn_init_performance[ori_method, is_outer_soft, max_iter]

    while (model_iter_count < max_iter)
        
        tStartIter = tic
        
        ## add counter
        model_iter_count = model_iter_count + 1  
            
        # ## create dir for current folder
        # top_out_folder = sprintf("#s/Iter_#d", out_dir, model_iter_count)
        # if (~exist(top_out_folder, "dir"))
        #     mkdir(top_out_folder)
        # end    
        # str = sprintf("*** Iter = #d:\n", model_iter_count)
        # CTHMM_print_log[str]
        
        # ## reset main global parameters
        # CTHMM_learn_iter_reset_variables[]
            
        ## do precomputation for each method
        CTHMM_learn_iter_precomputation[]
        num_distinct_time = size(distinct_time_list, 1)
            
        ## batch outer soft/hard decoding
        [cur_all_subject_prob] = CTHMM_learn_batch_outer_decoding_Etij_for_subjects[is_outer_soft, train_idx_list]    
        
        ## now we can start learning for every distinct time
        tStartTemp = tic()
        str = sprintf("For each distinct time interval, compute inner expectations...\n")
        fprintf(str)
        CTHMM_learn_Expm_accum_Nij_Ti_for_all_time_intervals[]

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