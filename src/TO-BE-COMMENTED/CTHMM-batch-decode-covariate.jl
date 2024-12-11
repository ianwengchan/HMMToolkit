# VERSION 2: WITH COVARIATES, using the no covariate functions
function CTHMM_batch_decode_Etij_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0, group_by_col = nothing)

    ## precomputation
    num_state = size(π_list, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_subject)
    Etij_list = Array{Array{Float64, 3}}(undef, num_subject)   # store Etij for each subject, i.e. a specific covariate combination
    # note that the length of t depends on the subject
    Svi_list = Array{Array{Float64}}(undef, num_subject)

    GC.safepoint()

    # cur_all_subject_prob = 0.0
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        Svi, subject_log_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df_n, response_list, Qn, π_list, state_list; ϵ = ϵ, group_by_col = group_by_col)
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob
        Svi_list[n] = Svi
        cur_all_subject_prob_list[n] = subject_log_prob
        Etij_list[n] = Etij
        GC.safepoint()
    end

    GC.safepoint()

    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    Svi_full = reduce(vcat, Svi_list)

    return Svi_full, cur_all_subject_prob, Etij_list
    
end


function CTHMM_batch_decode_Etij_and_append_Svi_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0, group_by_col = nothing)

    Svi_full, cur_all_subject_prob, Etij_list = CTHMM_batch_decode_Etij_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = ϵ, group_by_col = group_by_col)

    num_state = size(π_list, 1)
    for i in 1:num_state
        df[:, string("Sv", i)] = Svi_full[:, i]
    end

    GC.safepoint()

    return cur_all_subject_prob, Etij_list

end


function CTHMM_batch_decode_for_cov_subjects_list(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0, group_by_col = nothing)

    # returning a list of list for each covariate subject and each of its trip
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    num_state = size(π_list, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    subject_trip_prob_list = Array{Array}(undef, num_subject)

    GC.safepoint()

    # cur_all_subject_prob = 0.0
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        subject_trip_prob_list[n] = CTHMM_batch_decode_for_subjects_list(soft_decode, df_n, response_list, Qn, π_list, state_list; ϵ = ϵ, group_by_col = group_by_col)
        GC.safepoint()
    end

    GC.safepoint()

    return subject_trip_prob_list
    
end


function CTHMM_batch_decode_for_cov_subjects(soft_decode, df, response_list, subject_df, covariate_list, α, π_list, state_list; ϵ = 0, group_by_col = nothing)

    # returning single value from all subjects and all trips
    # batch decoding and compute loglikelihood ONLY
    # WITHOUT producing Etij matrix NOR updating Svi

    ## precomputation
    num_state = size(π_list, 1)

    group_df = groupby(df, :SubjectId)
    num_subject = size(group_df, 1)
    cur_all_subject_prob_list = Array{Float64}(undef, num_subject)

    GC.safepoint()

    # cur_all_subject_prob = 0.0
    @threads for n = 1:num_subject
        df_n = group_df[n]
        Qn = HMMToolkit.build_cov_Q(num_state, α, hcat(subject_df[n, covariate_list]...))
        subject_log_prob = CTHMM_batch_decode_for_subjects(soft_decode, df_n, response_list, Qn, π_list, state_list; ϵ = ϵ, group_by_col = group_by_col)
        # cur_all_subject_prob = cur_all_subject_prob + subject_log_prob
        cur_all_subject_prob_list[n] = subject_log_prob
        GC.safepoint()
    end

    GC.safepoint()

    cur_all_subject_prob = sum(cur_all_subject_prob_list)

    return cur_all_subject_prob
    
end