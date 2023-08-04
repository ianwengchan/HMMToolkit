function CTHMM_precompute_batch_data_emission_prob(df)  # works but need to adjust distributions
        
    ## data emission probability
    group_df = groupby(df, :ID)
    
    num_time_series = size(group_df, 1)
    num_state = 4
    # num_state = size(state_list, 1) # what is the format of state list??

    obs_seq_list = Array{Any}(undef, num_time_series)
    
    for g = 1:num_time_series   # number of time series to consider, e.g. trips

        len_time_series = nrow(group_df[g]) - 1

        obs_seq_list[g] = Array{Any}(undef, 2)
        obs_seq_list[g][1] = zeros(Union{Float64,Missing}, len_time_series, num_state)  # data_emiss_prob_list
    
        ## compute emission probabilities for each dimension of observations for each state
        ## assume each dimension of observations are independent conditioned on the state
        data = group_df[g].delta_radian # just as an example

        for s = 1:num_state
            emiss_prob = pdf.(Normal(0, (0.5*s)), skipmissing(data))    # testing with variances 0.5, 1, 1.5, 2
            # emiss_prob = mvnpdf[data, state_list[s].mu, state_list[s].var]  # assume normal at this point; change probability distributions later
            
            obs_seq_list[g][1][:, s] = [missing; emiss_prob]
        end

        obs_seq_list[g][2] = log.(obs_seq_list[g][1])  # log_data_emiss_prob_list
        
    end

    return obs_seq_list

end



function CTHMM_learn_onetime_precomputation(train_idx_list)

    # global is_use_distinct_time_grouping
    # global distinct_time_list
    # global learn_performance
    
    ## precomputation of state emission prob of observations
    ## find out distinct time from the dataset
    distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)

    if (is_use_distinct_time_grouping .== 1)
        [distinct_time_list] = CTHMM_precompute_distinct_time_intv[train_idx_list]
    end
    ## compute all data emission probability
    CTHMM_precompute_batch_data_emission_prob[train_idx_list]

end