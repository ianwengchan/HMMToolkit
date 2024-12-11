"""
    CTHMM_learn_update_Q_mat(Nij_mat, taui_list; block_size = nothing)

Updates transition rate matrix `Q_mat`,
given number of state transition `nij`,
and duration in each state `taui`;
with the option that transition probability matrix is a block matrix defined by `block_size`.

# Arguments
- `Nij_mat`: A matrix of number of transitions between states i to j.
- `taui_list`: A list of duration in each state i.

# Optional Arguments
- `block_size`: A vector specifying the structure of the trasition probability matrix; default to nothing (not a block matrix).

# Return Values
An updated transition rate matrix `Q_mat`.
"""

function CTHMM_learn_update_Q_mat(Nij_mat, taui_list; block_size = nothing)
    
    num_state = size(Nij_mat, 1)
    Q_mat = zeros(num_state, num_state)

    for i = 1:(num_state-1)
        temp = Nij_mat[i, :] ./ taui_list[i]    # check if denominator is nonzero
        Q_mat[i, :] = replace!(temp, Inf => 0.0)
        Q_mat[i, i] = -(sum(temp) - temp[i])    # sum the qij terms
    end

    if (isnothing(block_size))
        Q_mat[num_state, :] .= 1
        Q_mat[num_state, num_state] = -(sum(Q_mat[num_state, :]) - 1)  # sum the qij terms
    else
        for i in 1:length(block_size)
            row = sum(block_size[1:i])
            Q_mat[row, :] .= 0
            Q_mat[row, (row - block_size[i] + 1):(row - 1)] .= 1
            Q_mat[row, row] = -(sum(Q_mat[row, :]))
        end
    end

    GC.safepoint()

    return Q_mat

end