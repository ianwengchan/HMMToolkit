# input
num_state = size(Q_mat, 1)
distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

function [Evij, log_prob, Pt_list] = CTHMM_decode_outer_forward_backward(seq_df)

    # decoding for only one time-series
    # even if i do decoding for all time-series
    # i only have to compute the distinct_time_list and distinct_time_Pt_list for once

    ## compute alpha()
    len_time_series = nrow(seq_df) - 1
    time_interval_list = seq_df.time_interval    # already feed-in the time intervals

    ## precompute data emission prob
    ## obs_seq_emiss_list[g][1] # data_emiss_prob_list
    ## obs_seq_emiss_list[g][2] # log_data_emiss_prob_list

    # if (obs_seq.has_compute_data_emiss_prob .== 0)
    #     for v = 1:len_time_series
    #         data = obs_seq.visit_list[v].data
    #         for m = 1:num_state            
    #             emiss_prob = mvnpdf[data, state_list[m].mu, state_list[m].sigma]
    #             obs_seq.data_emiss_prob_list[v,m] = emiss_prob; 
    #             #obs_seq.log_data_emiss_prob_list[v,m] = log(emiss_prob)
    #         end
    #     end
    # end

    ALPHA = zeros(Union{Float64,Missing}, len_time_series, num_state)
    C = zeros(Union{Float64,Missing}, num_state, 1) # rescaling factor

    ## precomputing of Pt for every timepoint
    Pt_list = Array{Any}(undef, len_time_series)
    for v = 1:len_time_series
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
    end

    ## init alpha for time 1
    ALPHA[1, :] = state_init_prob_list .* data_emiss_prob_list[1, :]
    C[1] = 1.0 / sum(ALPHA[1, :])
    ALPHA[1, :] = ALPHA[1, :] .* C[1]
    
    for v = 2:len_time_series

        for s = 1:num_state # current state
            prob = 0.0
            prob = prob + sum(ALPHA[v-1, :] .* Pt_list[v-1][:, s]) * obs_seq.data_emiss_prob_list[v, s]
            ALPHA[v, s] = prob
            # for k = 1:num_state # for each previous state                       
            #    prob = prob + ALPHA[v-1, k] * Pt_list[v-1](k, s)
            # end                
            # prob = prob * obs_seq.data_emiss_prob_list[v, s]
            # ALPHA[v, s] = prob;               
        end # s

        # scaling
        C[v] = 1.0 / sum(ALPHA[v, :])
        ALPHA[v, :] = ALPHA[v, :] .* C[v]
    end # v

    
    ## compute beta()
    BETA = zeros(len_time_series, num_state)

    # init beta()
    BETA[len_time_series, :] .= C[len_time_series]
    
    for v = (len_time_series):(-1):1

        for s = 1:num_state # current state
            prob = 0.0
            prob = prob + sum(Pt_list[v][s, :] .* obs_seq.data_emiss_prob_list[v+1, :] .* BETA[v+1, :])
            BETA[v, s] = prob
        end # s

        BETA[v, :] = BETA[v, :] .* C[v]
    end # v


    ## compute Evij
    Evij = zeros(len_time_series, num_state, num_state)

    for v = 1:len_time_series

        for i = 1:num_state
            for j = 1:num_state
                Evij[v, i, j] = ALPHA[v, i] * Pt_list[v][i, j] * obs_seq.data_emiss_prob_list[v+1, j] * BETA[v+1, j]
            end
        end
        sum_Evij = sum(Evij[v, :, :])
        Evij[v, :, :] = Evij[v, :, :] / sum_Evij
    
    end
    
    
    ## check rabinar's paper
    log_prob = 0
    log_prob = log_prob - sum(log.(C))
    
end