function CTHMM_learn_update_Q_mat(Nij_mat, taui_list)
    
    num_state = size(Nij_mat, 1)
    Q_mat = zeros(num_state, num_state)

    for i = 1:num_state
        temp = Nij_mat[i, :] ./ taui_list[i]    # check if denominator is nonzero
        Q_mat[i, :] = replace!(temp, Inf => 0.0)
        Q_mat[i, i] = -(sum(temp) - temp[i])    # sum the qij terms
    end

    ## output some statistics
    min_qii = minimum(Q_mat)    # qii <= 0
    max_qij = maximum(Q_mat)    # qij >= 0

    println("min_qii = ", min_qii)
    println("max_qij = ", max_qij)

    return Q_mat

end