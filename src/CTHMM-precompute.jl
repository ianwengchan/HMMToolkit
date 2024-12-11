"""
    CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list; group_by_col = nothing)

Computes the data emission probability in batches,
given dataframe `df`,
list of responses `response_list`,
list of state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `df`: Dataframe of (multiple) time series, identified by "ID".
- `response_list`: List of responses to compute the data emission probability.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- An array of g list of data emission probabilities for g time series;
    for each time series, returns 2 matrices, each of (len_time_series)-by-(num_state),
    being data_emiss_prob_list and its logged version, respectively.
"""

function CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list; group_by_col = nothing)
        
    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end
    
    num_time_series = size(group_df, 1)
    num_state = size(state_list, 2) # follow the expert distribution list format of LRMoE
    num_dim = size(state_list, 1)

    obs_seq_emiss_list = Array{Array{Matrix{Float64}}}(undef, num_time_series)

    GC.safepoint()
    
    @threads for g = 1:num_time_series   # number of time series to consider, e.g. trips

        len_time_series = nrow(group_df[g])
        # observations can be missing, e.g. speed is available starting 1st timepoint
        # acceleration and angle are available starting 2nd timepoint
        # but change in angle is available only starting 3rd timepoint

        obs_seq_emiss_list[g] = Array{Matrix{Float64}}(undef, 2)
        obs_seq_emiss_list[g][1] = zeros(len_time_series, num_state)  # data_emiss_prob_list
    
        ## compute emission probabilities for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = select(group_df[g], response_list)

        for s = 1:num_state
            temp = 1.0
            for d = 1:num_dim
                # this is the correct pdf for ZI distributions
                temp = temp .* map(x -> ismissing(x) ? 1 : max(1e-99, exp.(HMMToolkit.expert_ll_exact.(state_list[d, s], x))), data[:, d])
                # assume emission probability of missing data = 1
                # floor the observation pdf
            end
            obs_seq_emiss_list[g][1][:, s] = temp
            GC.safepoint()
        end

        obs_seq_emiss_list[g][2] = log.(obs_seq_emiss_list[g][1])  # log_data_emiss_prob_list

        GC.safepoint()
        
    end

    GC.safepoint()

    return obs_seq_emiss_list

end



"""
    CTHMM_precompute_batch_data_emission_prob_separate(df, response_list, state_list; group_by_col = nothing)

Computes the data emission probability separately (for pseudo-residuals) in batches, 
given dataframe `df`,
list of responses `response_list`,
list of state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `df`: Dataframe of (multiple) time series, identified by "ID".
- `response_list`: List of responses to compute the data emission probability.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- An array of g list of data emission probabilities for g time series;
    for each time series, returns (num_dim) matrices, each of (len_time_series)-by-(num_state),
    being data_emiss_prob_list for each dimension separately, respectively.
"""

function CTHMM_precompute_batch_data_emission_prob_separate(df, response_list, state_list; group_by_col = nothing)
        
    ## data emission pdf; for forecast pseudo-residuals
    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end
    
    num_time_series = size(group_df, 1)
    num_state = size(state_list, 2) # follow the expert distribution list format of LRMoE
    num_dim = size(state_list, 1)

    obs_seq_emiss_list = Array{Array{Matrix{Float64}}}(undef, num_time_series)

    GC.safepoint()
    
    @threads for g = 1:num_time_series   # number of time series to consider, e.g. trips

        len_time_series = nrow(group_df[g])
        # observations can be missing, e.g. speed is available starting 1st timepoint
        # acceleration and angle are available starting 2nd timepoint
        # but change in angle is available only starting 3rd timepoint

        obs_seq_emiss_list[g] = Array{Matrix{Float64}}(undef, num_dim)
        
        ## compute emission pdf SEPARATELY for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = select(group_df[g], response_list) # just as an example
        
        for d = 1:num_dim
            obs_seq_emiss_list[g][d] = zeros(len_time_series, num_state)  # data_emiss_pdf_list
        
            for s = 1:num_state
                temp = 1.0
                temp = temp .* map(x -> ismissing(x) ? 1 : max(1e-99, exp.(HMMToolkit.expert_ll_exact.(state_list[d, s], x))), data[:, d])
                # assume emission probability of missing data = 1
                # floor the observation pdf
                
                obs_seq_emiss_list[g][d][:, s] = temp
            end
            GC.safepoint()
            
        end

        GC.safepoint()
        
    end

    GC.safepoint()

    return obs_seq_emiss_list

end



"""
    CTHMM_precompute_batch_data_emission_cdf_separate(df, response_list, state_list; group_by_col = nothing)

Computes the data emission probability (cdf) separately (for pseudo-residuals) in batches, 
given dataframe `df`,
list of responses `response_list`,
list of state dependent distributions `state_list`;
with the option to be grouped by a list of columns in the dataframe `group_by_col`.

# Arguments
- `df`: Dataframe of (multiple) time series, identified by "ID".
- `response_list`: List of responses to compute the data emission probability.
- `state_list`: List of state dependent distributions corresponding to `response_list`.

# Optional Arguments
- `group_by_col`: List of columns to group the dataframe; if nothing is provided, group by "ID" by default.

# Return Values
- An array of g list of data emission cdf for g time series;
    for each time series, returns (num_dim) matrices, each of (len_time_series)-by-(num_state),
    being data_emiss_cdf_list for each dimension separately, respectively.
"""

function CTHMM_precompute_batch_data_emission_cdf_separate(df, response_list, state_list; group_by_col = nothing)
        
    ## data emission cdf; for forecast pseudo-residuals
    if isnothing(group_by_col)
        group_df = groupby(df, :ID)
    else
        group_df = groupby(df, group_by_col)
    end
    
    num_time_series = size(group_df, 1)
    num_state = size(state_list, 2) # follow the expert distribution list format of LRMoE
    num_dim = size(state_list, 1)

    obs_seq_emiss_list = Array{Array{Matrix{Float64}}}(undef, num_time_series)

    GC.safepoint()
    
    @threads for g = 1:num_time_series   # number of time series to consider, e.g. trips

        len_time_series = nrow(group_df[g])
        # observations can be missing, e.g. speed is available starting 1st timepoint
        # acceleration and angle are available starting 2nd timepoint
        # but change in angle is available only starting 3rd timepoint

        obs_seq_emiss_list[g] = Array{Matrix{Float64}}(undef, num_dim)
        
        ## compute emission cdf SEPARATELY for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = select(group_df[g], response_list) # just as an example
        
        for d = 1:num_dim
            obs_seq_emiss_list[g][d] = zeros(len_time_series, num_state)  # data_emiss_cdf_list
        
            for s = 1:num_state
                temp = 1.0
                temp = temp .* map(x -> ismissing(x) ? 1 : max(1e-99, HMMToolkit.cdf.(state_list[d, s], x)), data[:, d])
                # fixed the cdf for ZI distributions, directly calling cdf function should be fine
                # assume emission probability of missing data = 1
                # floor the observation cdf
                
                obs_seq_emiss_list[g][d][:, s] = temp
            end
            GC.safepoint()
            
        end

        GC.safepoint()
        
    end

    GC.safepoint()

    return obs_seq_emiss_list

end