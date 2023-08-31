using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2, StatsBase, Bessels, Random
using .CTHMM

# using XLSX
# xf = DataFrame(XLSX.readtable(datadir("test_set.xlsx"), "Sheet1"))

include(srcdir("CTHMM-precompute-distinct-time.jl"))
include(srcdir("CTHMM-precompute.jl"))
include(srcdir("CTHMM-decode-forward-backward.jl"))
include(srcdir("CTHMM-decode-viterbi.jl"))
include(srcdir("CTHMM-batch-decode.jl"))
include(srcdir("CTHMM-learn-nij-taui.jl"))
include(srcdir("CTHMM-learn-Q.jl"))
include(srcdir("CTHMM-learn-EM.jl"))
include(srcdir("CTHMM-simulation.jl"))

df_longer = load(datadir("df_longer.jld2"), "df_longer")

# test with a CTHMM with 3 states, initiate Q
Q_mat0 = [-0.6 0.2 0.4;
        0.4 -1 0.6;
        0.25 0.5 -0.75]

π_list0 = [0.5; 0.3; 0.2]

response_list = ["delta_radian", "acceleration"]

# state_list0 = [CTHMM.VonMisesExpert(0, 1.5) CTHMM.VonMisesExpert(-0.5, 1.5) CTHMM.VonMisesExpert(0.5, 1.5);
#                 CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(-1.5, 1) CTHMM.NormalExpert(1.5, 1)]

state_list0 = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-5, 1) CTHMM.NormalExpert(5, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(2, 2) CTHMM.VonMisesExpert(-2, 2)]


# Q_mat0 = [-1.5 0.5 0.5 0.5;
#         0.1 -0.3 0.1 0.1;
#         0.2 0.2 -0.6 0.2;
#         0.5 0.5 0.5 -1.5]

# π_list0 = [0.5; 0.25; 0.15; 0.1]

# state_list_init = [CTHMM.NormalExpert(0, 0.5) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1.5) CTHMM.NormalExpert(0, 2);
#                 CTHMM.NormalExpert(1, 0.5) CTHMM.NormalExpert(1, 1) CTHMM.NormalExpert(1, 1.5) CTHMM.NormalExpert(1, 2)]


# df = Base.copy(df_longer)
# Q_mat_init = Base.copy(Q_mat0)
# π_list_init = Base.copy(π_list0)
# state_list_init = Base.copy(state_list0)

# fitted = CTHMM_learn_EM(df, response_list, Q_mat_init, π_list_init, state_list_init)


Q_mat0
π_list0
state_list0

df_sim = sim_dataset(Q_mat0, π_list0, state_list0, 500)

Q_mat_init = [-0.2 0.1 0.1;
        0.2 -0.4 0.2;
        0.3 0.3 -0.6]

π_list_init = [0.3; 0.3; 0.4]

# state_list_init = [CTHMM.VonMisesExpert(0, 1) CTHMM.VonMisesExpert(0, 1) CTHMM.VonMisesExpert(0, 1);
#                 CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-5, 1) CTHMM.NormalExpert(5, 1);
                        CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(2, 2) CTHMM.VonMisesExpert(-2, 2)]


obs_seq_emiss_list = CTHMM_precompute_batch_data_emission_prob(df_sim, ["response1"], state_list0)

fitted = CTHMM_learn_EM(df_sim, ["response1", "response2"], Q_mat_init, π_list_init, state_list0; max_iter = 1000, Q_max_iter = 5)

fitted.Q_mat_fit

# one dimension NormalExperts
# [-0.613577   0.166676   0.446901;
# 0.41245   -0.936028   0.523578;
# 0.24946    0.49158   -0.741039]
# converged at 803 iterations

# two dimensions, NormalExperts and VonMisesExperts
# [-0.614169   0.24876    0.365409;
# 0.434089  -0.949363   0.515274;
# 0.247857   0.414409  -0.662266]
# converged at 620 iterations