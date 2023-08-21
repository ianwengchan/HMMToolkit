using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2
using .CTHMM

# CTHMM-precompute-distinct-time.jl - tested
# CTHMM-precompute.jl - partially tested
# CTHMM-decode-forward-backward.jl - tested
# CTHMM-decode-viterbi.jl - tested
# CTHMM-batch-decode.jl - tested
# CTHMM-learn-nij-taui.jl - tested
# CTHMM-learn-Q.jl - tested
# CTHMM-

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

state_init_prob_list0 = [0.5; 0.25; 0.15; 0.1]

response_list = ["delta_radian", "acceleration"]

df = Base.copy(df_longer)
Q_mat_init = Base.copy(Q_mat0)
state_init_prob_list_init = Base.copy(state_init_prob_list0)

state_list_init = [CTHMM.NormalExpert(0, 0.5) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1.5) CTHMM.NormalExpert(0, 2);
                CTHMM.NormalExpert(1, 0.5) CTHMM.NormalExpert(1, 1) CTHMM.NormalExpert(1, 1.5) CTHMM.NormalExpert(1, 2)]

distinct_time_list = CTHMM_precompute_distinct_time_list(df.time_interval)
num_state = size(Q_mat_init, 1)
num_dim = size(state_list_init, 1)
num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series
# have to redo for updated Q and distribution parameters:
distinct_time_Pt_list = CTHMM_precompute_distinct_time_Pt_list(distinct_time_list, Q_mat_init)  # with initial guess Q0
obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df, response_list, state_list_init)  # with initial parameter guess

pre_all_subject_prob = -Inf
model_iter_count = 0

Q_mat = Base.copy(Q_mat_init)
state_init_prob_list = Base.copy(state_init_prob_list_init)
state_list = Base.copy(state_list_init)

model_iter_count = model_iter_count + 1

cur_all_subject_prob, Etij = CTHMM_batch_decode_Etij_for_subjects(1, df, response_list, Q_mat, state_init_prob_list, state_list)

Nij_mat, taui_list = CTHMM_learn_nij_taui(distinct_time_list, distinct_time_Pt_list, Q_mat, Etij)
Q_mat_new = CTHMM_learn_update_Q_mat(Nij_mat, taui_list)

for d in 1:num_dim
        for i in 1:num_state
                state_list[d, i] = CTHMM.EM_M_expert_exact(state_list[d, i], df[:, response_list[d]], df[:, string("Sv", i)])
        end
end

state_list


