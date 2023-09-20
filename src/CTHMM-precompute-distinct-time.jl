function CTHMM_precompute_distinct_time_list(time_interval_list)    # tested
    
    ## find and sort all distinct time intervals in the dataset
    ## only need to compute once throughout the fitting
    distinct_time_list = sort(unique(skipmissing(time_interval_list)))
        
    return distinct_time_list
end


function CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)  # tested

    num_distinct_time = size(distinct_time_list, 1)
    distinct_time_Pt_list = Array{Matrix{Float64}}(undef, num_distinct_time)

    # for each distinct time
    @threads for t_idx = 1:num_distinct_time
            
        T = distinct_time_list[t_idx]   # t_delta
        distinct_time_Pt_list[t_idx] = exp(Q_mat * T)

    end

    return distinct_time_Pt_list
        
end


function CTHMM_precompute_distinct_time_log_Pt_list(distinct_time_list, Q_mat)  # tested

    num_distinct_time = size(distinct_time_list, 1)
    distinct_time_log_Pt_list = Array{Matrix{Float64}}(undef, num_distinct_time)

    # for each distinct time
    @threads for t_idx = 1:num_distinct_time
            
        T = distinct_time_list[t_idx]   # t_delta
        distinct_time_log_Pt_list[t_idx] = log.(exp(Q_mat * T))

    end

    return distinct_time_log_Pt_list
        
end