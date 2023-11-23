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

        Ri_list = Array{Float64}(undef, num_distinct_time)
        
        @threads for t_idx = 1:num_distinct_time # delta = 1:r

            T = distinct_time_list[t_idx]   # t_delta

            expm_A = exp(A * T)[1:num_state, (num_state+1):end]

            Ri = 0.0
            temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]  # check if denominator is nonzero
            Ri = Ri + sum(filter(x -> x != Inf, temp))

            Ri_list[t_idx] = Ri
            GC.safepoint()
        end

        # tau_i
        taui_list[i] = taui_list[i] + sum(Ri_list)    # storing 1 tau_i for each i from all delta

        # N_ii
        ni = sum(Ri_list) * (-Q_mat[i, i])
        Nij_mat[i, i] = Nij_mat[i, i] + ni  # same A for calculating N_ii
        
        ## restore
        A[i, i + num_state] = 0

        GC.safepoint()
        
    end
            
    
    # for each link; compute n_ij
    for i = 1:num_state
        
        for j = 1:num_state
            
            if (i != j) # the case of i,i is computed above
                            
                ## set up A
                A[i, j + num_state] = 1

                temp_sum_list = Array{Float64}(undef, num_distinct_time)
                
                @threads for t_idx = 1:num_distinct_time
                    
                    T = distinct_time_list[t_idx]   # t_delta

                    expm_A = exp(A * T)[1:num_state, (num_state+1):end]
                                    
                    temp_sum = 0.0
                    temp = Etij[t_idx, :, :] .* expm_A ./ distinct_time_Pt_list[t_idx]
                    temp_sum = temp_sum + sum(filter(x -> x != Inf, temp))

                    temp_sum_list[t_idx] = temp_sum
                    GC.safepoint()
                end
    
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

    return Nij_mat, taui_list
end




function CTHMM_learn_cov_nij_taui(num_state, num_subject, subject_df, covariate_list, distinct_time_list, α, Etij_list)

    Nij_list_all = Array{Array{Float64}}(undef, num_subject)
    taui_list_all = Array{Array{Float64}}(undef, num_subject)

    @threads for n = 1:num_subject
        Qn = CTHMM.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list[n], Qn)
        Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list[n], distinct_time_Pt_list, Qn, Etij_list[n])
        Nij_list_all[n] = Nij_mat
        taui_list_all[n] = taui_list
        GC.safepoint()
        # for i = 1:num_state
        #     subject_df[n, string("tau", i)] = taui_list[i]
        #     for j = 1:num_state
        #         subject_df[n, string("N", i, j)] = Nij_mat[i, j]
        #     end
        # end
    end

    for i = 1:num_state
        subject_df[:, string("tau", i)] = map(x -> x[i], taui_list_all)
        for j = 1:num_state
            subject_df[:, string("N", i, j)] = map(x -> x[i, j], Nij_list_all)
        end
    end

    GC.safepoint()

    return subject_df   # stored in subject_df

end