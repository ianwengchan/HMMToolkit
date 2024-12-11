function CTHMM_learn_cov_nij_taui(num_state, num_subject, subject_df, covariate_list, distinct_time_list, α, Etij_list)

    Nij_list_all = Array{Array{Float64}}(undef, num_subject)
    taui_list_all = Array{Array{Float64}}(undef, num_subject)

    GC.safepoint()

    @threads for n = 1:num_subject
        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
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

    GC.safepoint()

    for i = 1:num_state
        subject_df[:, string("tau", i)] = map(x -> x[i], taui_list_all)
        for j = 1:num_state
            subject_df[:, string("N", i, j)] = map(x -> x[i, j], Nij_list_all)
        end
    end

    GC.safepoint()

    return subject_df   # stored in subject_df

end