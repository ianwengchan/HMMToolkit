function CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)

    num_state = size(Q_mat, 1)
    num_distinct_time = size(distinct_time_list, 1)

    taui_list = zeros(num_state, 1)
    Nij_mat = zeros(num_state, num_state)
    
    ## construct A matrix
    A = zeros(num_state * 2, num_state * 2)
    A[1:num_state, 1:num_state] = Q_mat
    A[(num_state+1):end, (num_state+1):end] = Q_mat

    ## for each state; compute tau_i
    for i = 1:num_state
        
        ## set up A
        A[i, i + num_state] = 1
        
        for t_idx = 1:num_distinct_time # delta = 1:r

            T = distinct_time_list[t_idx]   # t_delta

            expm_A = exp(A * T)[1:num_state, (num_state+1):end]

            Ri = 0.0
            temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]  # check if denominator is nonzero
            Ri = Ri + sum(filter(x -> x != Inf, temp))

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

                    expm_A = exp(A * T)[1:num_state, (num_state+1):end]
                                    
                    temp_sum = 0.0
                    temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]
                    temp_sum = temp_sum + sum(filter(x -> x != Inf, temp))
    
                    nij = temp_sum * Q_mat[i, j]
    
                    ## accumulate count to Nij matrix
                    Nij_mat[i, j] = Nij_mat[i, j] + nij
                end
                
                ## restore    
                A[i, j + num_state] = 0
                
            end
        end
    end

    return Nij_mat, taui_list
end