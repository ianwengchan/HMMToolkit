function sim_time_series(ID, distinct_time_Pt_list, π_list, state_list)
    num_state = size(state_list, 2)
    num_dim = size(state_list, 1)
    len_time_series = rand(30:150)
    time_interval = rand(1:30, (len_time_series-1))
    state_tracker = zeros(len_time_series, num_state)
    
    state_tracker[1, :] = rand(Distributions.Multinomial(1, π_list))

    for v = 2:len_time_series
        prob = collect((state_tracker[(v-1), :]' * distinct_time_Pt_list[time_interval[v-1]])')
        state_tracker[v, :] = rand(Distributions.Multinomial(1, prob))
    end

    dff = DataFrame(ID = ID,
                    start = [1; fill(0, (len_time_series-1))],
                    time_interval = [missing; time_interval],
                    true_state = findfirst.(state_tracker[i, :] .== 1.0 for i in 1:len_time_series)
    )

    Y = vcat([CTHMM.sim_expert.(state_list) * state_tracker[i, :] for i in 1:len_time_series]'...)

    for i in 1:num_dim
        dff[:, string("response", i)] = Y[:, i]
    end

    return dff
end

function sim_dataset(Q_mat, π_list, state_list, num_time_series)
    # num_state = size(Q_mat, 1)
    # num_dim = size(state_list, 1)
    distinct_time_list = collect(1:1:30)
    distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    df_sim = sim_time_series(1, distinct_time_Pt_list, π_list, state_list)

    for g = 2:num_time_series

        dff = sim_time_series(g, distinct_time_Pt_list, π_list, state_list)
        df_sim = vcat(df_sim, dff)
    
    end

    return df_sim
end

function sim_dataset_Qn(α, subject_df, covariate_list, π_list, state_list, num_time_series)
    num_state = size(π_list, 1)
    num_subject = nrow(subject_df)
    
    Qn = CTHMM.build_cov_Q(num_state, α, hcat(subject_df[1, covariate_list]...))
    df_sim = sim_dataset(Qn, π_list, state_list, num_time_series)
    df_sim.SubjectId .= subject_df.SubjectId[1]

    for n = 2:num_subject

        Qn = CTHMM.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        df_n = sim_dataset(Qn, π_list, state_list, num_time_series)
        df_n.SubjectId .= subject_df.SubjectId[n]
        df_sim = vcat(df_sim, df_n)
    
    end

    return df_sim
end