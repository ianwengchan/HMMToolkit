function CTHMM_learn_update_Q_mat(taui_list, Nij_mat)
    
    num_state = size(Nij, 1)
    Q_mat = zeros(num_state, num_state)
    
    for i = 1:num_state
        for j = 1:num_state
             if (i != j)    # consider i,j entries where i != j
                 if (taui_list[i] > 0.0 && Nij_mat[i, j] .> 0.0)
                    Q_mat[i, j] = Nij_mat[i, j] / taui_list[i]
                 else()
                    Q_mat[i, j] = 0.0
                 end
             end
        end
        Q_mat[i, i] = -(sum(Q_mat[i, :]))   # consider i,i entries
    end
end

    # ## output some statistics
    # D = diag(Q_mat)
    # min_qii = minimum(D)
    # temp_Q = Q_mat
    # for i = 1:num_state
    #     temp_Q[i,i] = 0
    # end
    # max_qij = maximum(temp_Q)
    
    # str = sprintf("max_qij = #f, min_qii = #f\n", max_qij, min_qii)
    # CTHMM_print_log[str]