using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2, StatsBase, Bessels, Random, StatsPlots
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

function sample_df_by_seed(seed, num_time_series, df)
        Random.seed!(seed)
        train_id = sample(unique(df.ID), num_time_series)
        return filter(row -> row.ID in train_id, df)
end        

df_train = sample_df_by_seed(1234, 150, df_longer)

trip = sample_df_by_seed(1234, 1, df_longer)

density(skipmissing(df_longer.radian))  # 6 modes
density(skipmissing(df_longer.delta_radian))    # 4 modes: -1.5, 0, 1.5, 3.14??
density(filter(x -> x <= 60, skipmissing(df_longer.time_interval)))     # 5 sec is the mode
density(skipmissing(df_longer.Speed))   # 3/4 modes: 0, 30, 40, 120
density(skipmissing(df_longer.acceleration))    # 0 is the mode
density(filter(x -> (x <= 3 || x >= -3), skipmissing(df_longer.acceleration)))    # 0 is the mode


density(skipmissing(df_train.radian))  # 4 modes
density(skipmissing(df_train.delta_radian))     # 4 modes: -1.5, 0, 1.5, 3.14??
density(filter(x -> (x != 0), skipmissing(df_train.delta_radian)))
density(filter(x -> (x > 0.25 || x < -0.25), skipmissing(df_train.delta_radian)))    # 4 modes: -1.5, 0, 1.5, 3.14??
density(filter(x -> x <= 60, skipmissing(df_train.time_interval)))     # 5 sec is the mode
density(skipmissing(df_train.Speed))   # 3/4 modes: 0, 30, 40, 120
density(skipmissing(df_train.acceleration))    # 0 is the mode

Q_mat_init = [-0.2 0.1 0.1;
        0.2 -0.4 0.2;
        0.3 0.3 -0.6]

π_list_init = [0.3; 0.3; 0.4]

response_list = ["time_interval", "delta_radian", "acceleration"]

state_list_init = [CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2);
                CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

fitted = CTHMM_learn_EM(df_train, response_list, Q_mat_init, π_list_init, state_list0; max_iter = 1000, Q_max_iter = 5, penalty = false)

fitted.Q_mat_fit
fitted.π_list_fit
fitted.state_list_fit


Q_mat_init = [-0.3 0.1 0.1 0.1;
        0.2 -0.6 0.2 0.2;
        0.3 0.3 -0.9 0.3;
        0.4 0.4 0.4 -1.2]

π_list_init = [0.25; 0.25; 0.25; 0.25]

response_list = ["time_interval", "delta_radian", "acceleration"]

state_list_init = [CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2);
                CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

fitted4 = CTHMM_learn_EM(df_train, response_list, Q_mat_init, π_list_init, state_list_init; max_iter = 1000, Q_max_iter = 5)

fitted4.Q_mat_fit
fitted4.π_list_fit
fitted4.state_list_fit[3, :]


Q_mat_init = [-0.3 0.1 0.1 0.05 0.05;
        0.2 -0.6 0.2 0.1 0.1;
        0.3 0.3 -0.9 0.15 0.15;
        0.4 0.4 0.2 -1.2 0.2;
        0.5 0.5 0.25 0.25 -1.5]

π_list_init = [0.2; 0.2; 0.2; 0.2; 0.2]

response_list = ["time_interval", "delta_radian", "acceleration"]

state_list_init = [CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1) CTHMM.LogNormalExpert(0, 1);
                CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2) CTHMM.VonMisesExpert(0, 2);
                CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

fitted5 = CTHMM_learn_EM(df_train, response_list, Q_mat_init, π_list_init, state_list_init; max_iter = 1000, Q_max_iter = 5)

fitted5.Q_mat_fit
fitted5.π_list_fit
fitted5.state_list_fit[3, :]



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

π/2