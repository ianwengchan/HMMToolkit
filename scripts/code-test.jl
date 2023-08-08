using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Printf, Distributions

# CTHMM-precompute-distinct-time.jl - tested
# CTHMM-precompute.jl - partially tested
# CTHMM-decode-forward-backward.jl - working
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
len_time_series = nrow(seq_df) - 1
time_interval_list = collect(skipmissing(seq_df.time_interval))

num_state = size(Q_mat, 1)
ALPHA = zeros(Union{Float64,Missing}, len_time_series, num_state)
C = zeros(Union{Float64,Missing}, num_state, 1)

Pt_list = Array{Any}(undef, len_time_series)
for v = 1:len_time_series
        T = time_interval_list[v]
        t_idx = findfirst(x -> x .== T, distinct_time_list)
        Pt_list[v] = Array{Any}(undef, num_state, num_state)
        Pt_list[v] = distinct_time_Pt_list[t_idx]
end

ALPHA[1, :] = state_init_prob_list .* data_emiss_prob_list[1, :]
C[1] = 1.0 / sum(ALPHA[1, :])
ALPHA[1, :] = ALPHA[1, :] .* C[1]

v = 2

s = 1

ALPHA[v-1, :] .* Pt_list[v-1][:, s]