"""
    CTHMM_precompute_distinct_time_list(time_interval_list)

Finds and sorts all distinct time intervals in the dataset.

# Arguments
- `time_interval_list`: Time interval list in the dataset.

# Return Values
- A sorted list distinct time interval.
"""

function CTHMM_precompute_distinct_time_list(time_interval_list)
    
    ## find and sort all distinct time intervals in the dataset
    ## only need to compute once throughout the fitting
    distinct_time_list = sort(unique(skipmissing(time_interval_list)))
        
    return distinct_time_list
end



"""
    CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

Computes a list of state transition probability matrices from time 0 to time t Pt for each distinct time interval t,
given `distinct_time_list`
and transition rate matrix `Q_mat`.

Note: Ordering follows from `distinct_time_list`.

# Arguments
- `distinct_time_list`: Sorted distinct time interval list.
- `Q_mat`: Transition rate matrix.

# Return Values
- A sorted list of distinct state transition probability matrices Pt.
"""

function CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

    num_distinct_time = size(distinct_time_list, 1)
    distinct_time_Pt_list = Array{Matrix{Float64}}(undef, num_distinct_time)

    GC.safepoint()

    # for each distinct time
    @threads for t_idx = 1:num_distinct_time
            
        T = distinct_time_list[t_idx]   # t_delta
        distinct_time_Pt_list[t_idx] = exp(Q_mat * T)
        GC.safepoint()

    end

    GC.safepoint()

    return distinct_time_Pt_list
        
end



"""
    CTHMM_precompute_distinct_time_log_Pt_list(distinct_time_list, Q_mat)

Computes a list of logged state transition probability matrices from time 0 to time t log(Pt() for each distinct time interval t,
given `distinct_time_list`
and transition rate matrix `Q_mat`.

Note: Ordering follows from `distinct_time_list`.

# Arguments
- `distinct_time_list`: Sorted distinct time interval list.
- `Q_mat`: Transition rate matrix.

# Return Values
- A sorted list of logged distinct state transition probability matrices log(Pt).
"""

function CTHMM_precompute_distinct_time_log_Pt_list(distinct_time_list, Q_mat)

    num_distinct_time = size(distinct_time_list, 1)
    distinct_time_log_Pt_list = Array{Matrix{Float64}}(undef, num_distinct_time)

    GC.safepoint()

    # for each distinct time
    @threads for t_idx = 1:num_distinct_time
            
        T = distinct_time_list[t_idx]   # t_delta
        distinct_time_log_Pt_list[t_idx] = log.(exp(Q_mat * T))
        GC.safepoint()

    end

    GC.safepoint()

    return distinct_time_log_Pt_list
        
end