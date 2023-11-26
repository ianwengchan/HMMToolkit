function penalty_params(state_list, pen_params)
    return sum(
        sum([
            [penalize(state_list[j, k], pen_params[j][k]) for k in 1:size(state_list)[2]] for
            j in 1:size(state_list)[1]
        ]),
    )
end