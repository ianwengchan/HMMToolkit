using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2, StatsBase, Bessels, Random
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


# test with a CTHMM with 3 states, initiate Q
Q_mat0 = [-0.6 0.2 0.4;
        0.4 -1 0.6;
        0.25 0.5 -0.75]

π_list0 = [0.5; 0.3; 0.2]

state_list0 = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-5, 1) CTHMM.NormalExpert(5, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(2, 2) CTHMM.VonMisesExpert(-2, 2)]

Random.seed!(1234)
df_sim = sim_dataset(Q_mat0, π_list0, state_list0, 500)


Q_mat_init = [-0.2 0.1 0.1;
        0.2 -0.4 0.2;
        0.3 0.3 -0.6]

π_list_init = [0.3; 0.3; 0.4]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-5, 1) CTHMM.NormalExpert(5, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(2, 2) CTHMM.VonMisesExpert(-2, 2)]


fitted = CTHMM_learn_EM_Q_only(df_sim, ["response1", "response2"], Q_mat_init, π_list_init, state_list0; max_iter = 1000, Q_max_iter = 5)

fitted.Q_mat_fit

