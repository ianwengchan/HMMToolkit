function CTHMM_precompute_distinct_time_list(time_interval_list)    # tested
    
    ## find and sort all distinct time intervals in the dataset
    ## only need to compute once throughout the fitting
    distinct_time_list = sort(unique(skipmissing(time_interval_list)))
        
    return distinct_time_list
end


function CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)  # tested

    num_distinct_time = size(distinct_time_list, 1)
    distinct_time_Pt_list = Array{Any}(undef, num_distinct_time)

    # if (is_outer_soft .== 0)
    #     distinct_time_log_Pt_list = cell(num_time, 1)
    # end

    # for each distinct time
    for t_idx = 1:num_distinct_time
            
        T = distinct_time_list[t_idx]   # t_delta
        distinct_time_Pt_list[t_idx] = exp(Q_mat * T)

    end
            
    # if (is_outer_soft .== 0)
    #     distinct_time_log_Pt_list[i] = log(distinct_time_Pt_list[i])
    # end

    return distinct_time_Pt_list
        
end