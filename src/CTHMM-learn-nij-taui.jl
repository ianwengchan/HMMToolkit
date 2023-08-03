function CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Ctij, taui_list, Nij_mat)

    num_state = size(Q_mat, 1)
    num_distinct_time = size(distinct_time_list, 1)
    
    ## construct A matrix
    A = zeros(num_state * 2, num_state * 2)
    A[1:num_state, 1:num_state] = Q_mat
    A[(num_state+1):end, (num_state+1):end] = Q_mat
    ### OK


    ## for each state; compute tau_i
    for i = 1:num_state
        
        ## set up A
        A[i, i + num_state] = 1
        
        for t_idx = 1:num_distinct_time # delta = 1:r

            T = distinct_time_list[t_idx]   # t_delta

            expm_A = expm(A * T)[1:num_state, (num_state+1):end]

            Ri = 0.0
            for k = 1:num_state # k,l are pairs of i,j; assume reachability
                for l = 1:num_state
                    if (distinct_time_Pt_list[t_idx][k,l] != 0) # check if denominator is nonzero
                        Ri = Ri + Ctij[t_idx, k, l] * expm_A[k, l] / distinct_time_Pt_list[t_idx][k, l]  
                    end
                end
            end

            # tau_i
            taui_list[i] = taui_list[i] + Ri    # storing 1 tau_i for each i from all delta

            # N_ii
            ni = Ri * (-Q_mat[i, i])
            Nij_mat[i, i] = Nij_mat[i, i] + ni  # same A for calculating N_ii
        
        end
        
        ## restore    
        A[i, i + num_state] = 0
        
    end
            
    
    # for each link; compute n_ij
    for i = 1:num_state
        
        for j = 1:num_state
            
            if (i != j) # the case of i,i is computed above
                            
                ## set up A
                A[i, j + num_state] = 1
                
                for t_idx = 1:num_distinct_time
                    
                    T = distinct_time_list[t_idx]   # t_delta

                    expm_A = expm(A * T)[1:num_state, (num_state+1):end]
                                    
                    temp_sum = 0.0
                    for k = 1:num_state
                        for l = 1:num_state                            
                            if (distinct_time_Pt_list[t_idx][k, l] != 0)                        
                                temp_sum = temp_sum + Ctij[t_idx, k, l] * expm_A[k, l] / distinct_time_Pt_list[t_idx][k, l] 
                            end
                        end
                    end
    
                    nij = temp_sum * Q_mat[i, j]
    
                    ## accumulate count to Nij matrix
                    Nij_mat[i, j] = Nij_mat[i, j] + nij
                end
                
                ## restore    
                A[i, j + num_state] = 0
                
            end
        end
    end
end