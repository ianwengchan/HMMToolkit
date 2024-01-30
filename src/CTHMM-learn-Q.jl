function CTHMM_learn_update_Q_mat(Nij_mat, taui_list)
    
    num_state = size(Nij_mat, 1)
    Q_mat = zeros(num_state, num_state)

    for i = 1:num_state
        temp = Nij_mat[i, :] ./ taui_list[i]    # check if denominator is nonzero
        Q_mat[i, :] = replace!(temp, Inf => 0.0)
        Q_mat[i, i] = -(sum(temp) - temp[i])    # sum the qij terms
    end

    GC.safepoint()

    # ## output some statistics
    # min_qii = minimum(Q_mat)    # qii <= 0
    # max_qij = maximum(Q_mat)    # qij >= 0

    # println("min qii = ", min_qii)
    # println("max qij = ", max_qij)

    return Q_mat

end



function CTHMM_learn_update_Q_mat_new(Nij_mat, taui_list)
    
    num_state = size(Nij_mat, 1)
    Q_mat = zeros(num_state, num_state)

    for i = 1:(num_state-1)
        temp = Nij_mat[i, :] ./ taui_list[i]    # check if denominator is nonzero
        Q_mat[i, :] = replace!(temp, Inf => 0.0)
        Q_mat[i, i] = -(sum(temp) - temp[i])    # sum the qij terms
    end

    Q_mat[num_state, :] .= 1
    Q_mat[num_state, num_state] = -(sum(Q_mat[num_state, :]) - 1)  # sum the qij terms

    GC.safepoint()

    # ## output some statistics
    # min_qii = minimum(Q_mat)    # qii <= 0
    # max_qij = maximum(Q_mat)    # qij >= 0

    # println("min qii = ", min_qii)
    # println("max qij = ", max_qij)

    return Q_mat

end