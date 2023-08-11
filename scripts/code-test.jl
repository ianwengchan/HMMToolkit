using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Printf, Distributions

# CTHMM-precompute-distinct-time.jl - tested
# CTHMM-precompute.jl - partially tested
# CTHMM-decode-forward-backward.jl - Evij calculation tested
# CTHMM-decode-viterbi.jl
# CTHMM-learn-batch-outer-decoding.jl
# have to check calculation of Ctij (or Etij) first

# test with a CTHMM with 4 states, initiate Q
Q_mat = [-1.5 0.5 0.5 0.5;
        0.1 -0.3 0.1 0.1;
        0.2 0.2 -0.6 0.2;
        0.5 0.5 0.5 -1.5]

state_init_prob_list = [0.5; 0.25; 0.15; 0.1]

time_interval_list = df_longer.time_interval

distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
# OK


distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
# OK


group_df = groupby(df_longer, :ID)

num_time_series = size(group_df, 1)
num_state = 4
    
obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df_longer)
# OK


g = 1
seq_df = group_df[g]
data_emiss_prob_list = obs_seq_emiss_list[g][1]

Evij, log_prob = CTHMM_decode_outer_forward_backward(seq_df, data_emiss_prob_list, Q_mat)

