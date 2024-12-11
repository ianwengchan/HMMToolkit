function sim_dataset_Qn(α, subject_df, covariate_list, π_list, state_list, num_time_series)
    num_state = size(π_list, 1)
    num_subject = nrow(subject_df)
    
    Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[1, covariate_list]...))
    df_sim = sim_dataset(Qn, π_list, state_list, num_time_series)
    df_sim.SubjectId .= subject_df.SubjectId[1]

    for n = 2:num_subject

        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        df_n = sim_dataset(Qn, π_list, state_list, num_time_series)
        df_n.SubjectId .= subject_df.SubjectId[n]
        df_sim = vcat(df_sim, df_n)
    
    end

    return df_sim
end