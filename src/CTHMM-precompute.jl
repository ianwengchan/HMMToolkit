function CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list)
        
    ## data emission probability
    group_df = groupby(df, :ID)
    
    num_time_series = size(group_df, 1)
    num_state = size(state_list, 2) # follow the expert distribution list format of LRMoE
    num_dim = size(state_list, 1)

    obs_seq_emiss_list = Array{Any}(undef, num_time_series)
    
    for g = 1:num_time_series   # number of time series to consider, e.g. trips

        len_time_series = nrow(group_df[g])
        # observations can be missing, e.g. speed is available starting 1st timepoint
        # acceleration and radian are available starting 2nd timepoint
        # but change in radian is available only starting 3rd timepoint

        obs_seq_emiss_list[g] = Array{Any}(undef, 2)
        obs_seq_emiss_list[g][1] = zeros(len_time_series, num_state)  # data_emiss_prob_list
    
        ## compute emission probabilities for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = select(group_df[g], response_list) # just as an example

        for s = 1:num_state
            temp = 1.0
            for d = 1:num_dim
                temp = temp .* map(x -> ismissing(x) ? 1 : CTHMM.pdf.(state_list[d, s], x), data[:, d])
                # assume emission probability of missing data = 1
            end
            obs_seq_emiss_list[g][1][:, s] = temp        
        end

        obs_seq_emiss_list[g][2] = log.(obs_seq_emiss_list[g][1])  # log_data_emiss_prob_list
        
    end

    return obs_seq_emiss_list

end