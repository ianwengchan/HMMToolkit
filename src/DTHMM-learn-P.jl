"""
    DTHMM_learn_update_P_mat(Eij_all; block_size = nothing)

Updates transition rate matrix `Q_mat`,
given number of state transition `nij`,
and duration in each state `taui`;
with the option that transition probability matrix is a block matrix defined by `block_size`.

# Arguments
- `Eij_all`: Soft count tables `Eij`, expected transition probability matrices.

# Optional Arguments
- `block_size`: A vector specifying the structure of the trasition probability matrix; default to nothing (not a block matrix).

# Return Values
An updated transition probability matrix `P_mat`.
"""

function DTHMM_learn_update_P_mat(Eij_all; block_size = nothing)
    
    num_state = size(Eij_all, 1)
    P_mat = zeros(num_state, num_state)

    for i = 1:num_state
        temp = Eij_all[i, :] ./ sum(Eij_all[i, :]) # check if denominator is nonzero
        P_mat[i, :] = replace!(temp, Inf => 0.0)
        if sum(P_mat[i, :]) == 0
            P_mat[i, i] = 1.0  # block matrix, diagonal is always non-empty
        end
    end

    if(!isnothing(block_size))
        num_block = length(block_size)
        tmp = zeros(num_state, num_state)
        for i in 1:num_block
            row = sum(block_size[1:i])
            start_idx = row - block_size[i] + 1
            tmp[(start_idx):(row), (start_idx):(row)] = P_mat[(start_idx):(row), (start_idx):(row)]
        end
        P_mat = tmp ./ sum(tmp, dims = 2)
    end

    GC.safepoint()

    return P_mat

end