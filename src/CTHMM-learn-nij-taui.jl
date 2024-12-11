"""
    CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)

Learns number of state transition `nij` and duration in each state `taui`, 
given sorted list of distinct time intervals `distinct_time_list`,
corresponding sorted list of transition probability matrices `distinct_time_Pt_list`,
transition rate matrix  `Q_mat`,
and soft count table `Etij`.

# Arguments
- `distinct_time_list`: Sorted distinct time interval list.
- `distinct_time_Pt_list`: A sorted list of distinct state transition probability matrices Pt.
- `Q_mat`: Transition rate matrix.
- `Etij`: Soft count tables, expected transition probability matrices, for each distinct time interval t.

# Return Values
- `Nij_mat`: A matrix of number of transitions between states i to j.
- `taui_list`: A list of duration in each state i.
"""

function CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)

    num_state = size(Q_mat, 1)
    num_distinct_time = size(distinct_time_list, 1)

    taui_list = zeros(num_state, 1)
    Nij_mat = zeros(num_state, num_state)
    
    ## construct A matrix
    A = zeros(num_state * 2, num_state * 2)
    A[1:num_state, 1:num_state] = Q_mat
    A[(num_state+1):end, (num_state+1):end] = Q_mat

    GC.safepoint()

    ## for each state; compute tau_i
    for i = 1:num_state
        
        ## set up A
        A[i, i + num_state] = 1

        Ri_list = Array{Float64}(undef, num_distinct_time)

        GC.safepoint()
        
        @threads for t_idx = 1:num_distinct_time # delta = 1:r

            T = distinct_time_list[t_idx]   # t_delta

            expm_A = exp(A * T)[1:num_state, (num_state+1):end]

            Ri = 0.0
            temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]  # check if denominator is nonzero
            Ri = Ri + sum(filter(x -> x != Inf, filter(!isnan, temp)))

            Ri_list[t_idx] = Ri
            GC.safepoint()
        end

        GC.safepoint()

        # tau_i
        taui_list[i] = taui_list[i] + sum(Ri_list)    # storing 1 tau_i for each i from all delta

        # N_ii
        ni = sum(Ri_list) * (-Q_mat[i, i])
        Nij_mat[i, i] = Nij_mat[i, i] + ni  # same A for calculating N_ii
        
        ## restore
        A[i, i + num_state] = 0

        GC.safepoint()
        
    end

    GC.safepoint()
            
    
    # for each link; compute n_ij
    for i = 1:num_state
        
        for j = 1:num_state
            
            if (i != j) # the case of i,i is computed above
                            
                ## set up A
                A[i, j + num_state] = 1

                temp_sum_list = Array{Float64}(undef, num_distinct_time)

                GC.safepoint()
                
                @threads for t_idx = 1:num_distinct_time
                    
                    T = distinct_time_list[t_idx]   # t_delta

                    expm_A = exp(A * T)[1:num_state, (num_state+1):end]
                                    
                    temp_sum = 0.0
                    temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]
                    temp_sum = temp_sum + sum(filter(x -> x != Inf, filter(!isnan, temp)))

                    temp_sum_list[t_idx] = temp_sum
                    GC.safepoint()
                end

                GC.safepoint()
    
                nij = sum(temp_sum_list) * Q_mat[i, j]
    
                ## accumulate count to Nij matrix
                Nij_mat[i, j] = Nij_mat[i, j] + nij
                
                ## restore
                A[i, j + num_state] = 0

                GC.safepoint()
                
            end
            GC.safepoint()
        end
        GC.safepoint()
    end

    GC.safepoint()

    return (Nij_mat = Nij_mat, taui_list = taui_list)
end