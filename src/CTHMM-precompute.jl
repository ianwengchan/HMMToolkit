function CTHMM_precompute_batch_data_emission_prob(df, state_list)  # works but need to adjust distributions
        
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
        # but change in radian is avilable only starting 3rd timepoint

        obs_seq_emiss_list[g] = Array{Any}(undef, 2)
        obs_seq_emiss_list[g][1] = zeros(len_time_series, num_state)  # data_emiss_prob_list
    
        ## compute emission probabilities for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = [transpose(group_df[g].delta_radian);
                transpose(group_df[g].acceleration)] # just as an example

        for s = 1:num_state
            temp = 1.0
            for d = 1:num_dim
                temp = temp .* map(x -> ismissing(x) ? 1 : CTHMM.pdf.(state_list[d, s], x), data[d, :])
            end
            obs_seq_emiss_list[g][1][:, s] = temp
            
            # assume emission probability of missing data = 1
            # testing with variances 0.5, 1, 1.5, 2
            # emiss_prob = mvnpdf[data, state_list[s].mu, state_list[s].var]  # assume normal at this point; change probability distributions later
        end

        obs_seq_emiss_list[g][2] = log.(obs_seq_emiss_list[g][1])  # log_data_emiss_prob_list
        
    end

    return obs_seq_emiss_list

end



# function CTHMM_learn_onetime_precomputation(train_idx_list)

#     # global is_use_distinct_time_grouping
#     # global distinct_time_list
#     # global learn_performance
    
#     ## precomputation of state emission prob of observations
#     ## find out distinct time from the dataset
#     distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)

#     if (is_use_distinct_time_grouping .== 1)
#         [distinct_time_list] = CTHMM_precompute_distinct_time_intv[train_idx_list]
#     end
#     ## compute all data emission probability
#     CTHMM_precompute_batch_data_emission_prob[train_idx_list]

# end