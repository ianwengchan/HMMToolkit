using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2

# CTHMM-precompute-distinct-time.jl - tested
# CTHMM-precompute.jl - partially tested
# CTHMM-decode-forward-backward.jl - tested
# CTHMM-decode-viterbi.jl - tested
# CTHMM-batch-decode.jl - tested
# CTHMM-learn-nij-taui.jl - tested
# CTHMM-learn-Q.jl - tested

include(srcdir("CTHMM-precompute-distinct-time.jl"))
include(srcdir("CTHMM-precompute.jl"))
include(srcdir("CTHMM-decode-forward-backward.jl"))
include(srcdir("CTHMM-decode-viterbi.jl"))
include(srcdir("CTHMM-batch-decode.jl"))
include(srcdir("CTHMM-learn-nij-taui.jl"))

df_longer = load(datadir("df_longer.jld2"), "df_longer")

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
log_data_emiss_prob_list = obs_seq_emiss_list[g][2]

# log_prob from soft decoding is better than that from hard decoding
Evij, log_prob = CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, state_init_prob_list)
# OK
best_state_seq, best_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, state_init_prob_list)
# OK


cur_all_subject_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(1, df_longer, Q_mat, state_init_prob_list)
# OK


Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
# OK

Nij_mat
taui_list

Q_mat_new = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

