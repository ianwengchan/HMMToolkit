"""
    DTHMM_sim_time_series(ID, P_mat, π_list, state_list)

Simulate a single regular time series.

# Arguments
- `ID`: Identifier of the time series.
- `P_mat`: Transition probability matrix.
- `π_list`: Initial state probabilities.
- `state_list`: State dependent distributions.

# Return Values
A dataframe of a regular time series,
    with number of observations between 30 to 150.
"""

function DTHMM_sim_time_series(ID, P_mat, π_list, state_list)
    num_state = size(state_list, 2)
    num_dim = size(state_list, 1)
    len_time_series = rand(30:150)
    time_interval = fill(1, (len_time_series-1))
    time_since = cumsum(time_interval)
    state_tracker = zeros(len_time_series, num_state)
    
    state_tracker[1, :] = rand(Distributions.Multinomial(1, π_list))

    for v = 2:len_time_series
        prob = collect((state_tracker[(v-1), :]' * P_mat)')
        state_tracker[v, :] = rand(Distributions.Multinomial(1, prob))
    end

    dff = DataFrame(ID = ID,
                    start = [1; fill(0, (len_time_series-1))],
                    time_interval = [missing; time_interval],
                    time_since = [0; time_since],
                    true_state = findfirst.(state_tracker[i, :] .== 1.0 for i in 1:len_time_series)
    )

    Y = vcat([HMMToolkit.sim_expert.(state_list) * state_tracker[i, :] for i in 1:len_time_series]'...)

    for i in 1:num_dim
        dff[:, string("response", i)] = Y[:, i]
    end

    return dff
end



"""
    DTHMM_sim_dataset(P_mat, π_list, state_list, num_time_series)

Simulate multiple regular time series.

# Arguments
- `P_mat`: Transition probability matrix.
- `π_list`: Initial state probabilities.
- `state_list`: State dependent distributions.
- `num_time_series`: Number of time series to be simulated.

# Return Values
A dataframe of mutliple regular time series, identified by `ID`,
    each with number of observations between 30 to 150.
"""

function DTHMM_sim_dataset(P_mat, π_list, state_list, num_time_series)

    df_sim = DTHMM_sim_time_series(1, P_mat, π_list, state_list)

    for g = 2:num_time_series

        dff = DTHMM_sim_time_series(g, P_mat, π_list, state_list)
        df_sim = vcat(df_sim, dff)
    
    end

    return df_sim
end