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
include(srcdir("CTHMM-learn-Q.jl"))

df_longer = load(datadir("df_longer.jld2"), "df_longer")

# test with a CTHMM with 4 states, initiate Q
Q_mat0 = [-1.5 0.5 0.5 0.5;
        0.1 -0.3 0.1 0.1;
        0.2 0.2 -0.6 0.2;
        0.5 0.5 0.5 -1.5]

state_init_prob_list = [0.5; 0.25; 0.15; 0.1]


# time_interval_list = df_longer.time_interval

# distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
# # OK

# distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)
# # OK


# group_df = groupby(df_longer, :ID)

# num_time_series = size(group_df, 1)
# num_state = 4
    
# obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df_longer)
# # OK


# g = 1
# seq_df = group_df[g]
# data_emiss_prob_list = obs_seq_emiss_list[g][1]
# log_data_emiss_prob_list = obs_seq_emiss_list[g][2]

# num_state = size(Q_mat, 1)
# len_time_series = nrow(seq_df)
# time_interval_list = collect(skipmissing(seq_df.time_interval))

# distinct_time_list = CTHMM_precompute_distinct_time_list(time_interval_list)
# distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat)

# # log_prob from soft decoding is better than that from hard decoding
# Svi, Evij, log_prob = CTHMM_decode_forward_backward(seq_df, data_emiss_prob_list, Q_mat, state_init_prob_list)
# # OK
# best_state_seq, best_log_prob = CTHMM_decode_viterbi(seq_df, log_data_emiss_prob_list, Q_mat, state_init_prob_list)
# # OK
# cur_all_subject_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(1, df_longer, Q_mat, state_init_prob_list)
# # OK


# Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
# # OK

# Q_mat_new = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)


df = copy(df_longer)
Q_mat_init = copy(Q_mat0)

distinct_time_list = CTHMM_precompute_distinct_time_list(df.time_interval)
num_state = size(Q_mat_init, 1)
num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series
# have to redo for updated Q and distribution parameters:
distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat_init)  # with initial guess Q0
obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df)  # with initial parameter guess

pre_all_subject_prob = -Inf
model_iter_count = 0

Q_mat = copy(Q_mat_init)

model_iter_count = model_iter_count + 1

cur_all_subject_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(1, df, Q_mat, state_init_prob_list)

Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
Q_mat_new = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

ye = df.delta_radian
Y_e_obs = ye
Y_sq_e_obs = ye .^ 2

z_e_obs = df.Sv1
term_zkz = z_e_obs
term_zkz_Y = (z_e_obs .* Y_e_obs)
term_zkz_Y_sq = (z_e_obs .* Y_sq_e_obs)

μ_new = sum(skipmissing(term_zkz_Y))[1] / sum(skipmissing(term_zkz))[1]
demominator = sum(skipmissing(term_zkz))[1]

numerator = sum(skipmissing(term_zkz_Y_sq))[1] - 2.0 * μ_new * sum(skipmissing(term_zkz_Y))[1] +
                (μ_new)^2 * sum(skipmissing(term_zkz))[1]

tmp = numerator / demominator
σ_new = sqrt(maximum([0.0, tmp]))
