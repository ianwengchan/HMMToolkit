using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2, StatsBase, Bessels, Random, Clustering
using .CTHMM

include(srcdir("CTHMM-precompute-distinct-time.jl"))
include(srcdir("CTHMM-precompute.jl"))
include(srcdir("CTHMM-decode-forward-backward.jl"))
include(srcdir("CTHMM-decode-viterbi.jl"))
include(srcdir("CTHMM-batch-decode.jl"))
include(srcdir("CTHMM-learn-nij-taui.jl"))
include(srcdir("CTHMM-learn-Q.jl"))
include(srcdir("CTHMM-learn-EM.jl"))
include(srcdir("CTHMM-simulation.jl"))


# test with a CTHMM with 3 states
Q_mat0 = [-0.1 0.02 0.08;
        0.2 -1 0.8;
        0.8 1.2 -2] ./ 1000

π_list0 = [0.5; 0.3; 0.2]

state_list0 = [CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(-4, 1) CTHMM.NormalExpert(6, 2);
        CTHMM.VonMisesExpert(0, 10) CTHMM.VonMisesExpert(3, 5) CTHMM.VonMisesExpert(-2, 5);
        CTHMM.LaplaceExpert(0, 5) CTHMM.LaplaceExpert(-4, 4) CTHMM.LaplaceExpert(6, 3)]

# state_list0 = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-5, 2) CTHMM.NormalExpert(5, 2)]   

Random.seed!(1234)
df_sim = sim_dataset(Q_mat0, π_list0, state_list0, 500)


Q_mat_init = [-0.2 0.1 0.1;
        0.2 -0.4 0.2;
        0.3 0.3 -0.6]

π_list_init = [0.3; 0.3; 0.4]

# a non-informative initial guess also works
state_list_init = [CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(0, 2);
        CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2);
        CTHMM.LaplaceExpert(0, 2) CTHMM.LaplaceExpert(0, 2) CTHMM.LaplaceExpert(0, 2)]

response_list = ["response1", "response2", "response3"]

# fitted = CTHMM_learn_EM_expert_only(df_sim, response_list, Q_mat0, π_list0, state_list_init; max_iter = 1000, Q_max_iter = 5)
fitted = CTHMM_learn_EM(df_sim, response_list, Q_mat_init, π_list_init, state_list_init; max_iter = 1000, Q_max_iter = 5)

fitted.Q_mat_fit * 1000
fitted.π_list_fit
fitted.state_list_fit

# cmm for parameter initial guess
kmeans(hcat(df_sim.response1), 1)

histogram(df_sim.response1)

