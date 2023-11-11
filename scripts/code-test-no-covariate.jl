using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2, StatsBase, Bessels, Random, Clustering, Base.Threads
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

df_train = filter(rows -> rows.ID <= 400, df_sim)
df_test = filter(rows -> rows.ID > 400, df_sim)

df = df_train

fitted = CTHMM_learn_EM(df_train, response_list, Q_mat_init, π_list_init, state_list_init; max_iter = 200, Q_max_iter = 5)

fitted.Q_mat_fit * 1000
fitted.π_list_fit
fitted.state_list_fit


# soft-decoding by forward-backward algorithm, 1
# hard-decoding by Viterbi decoding, 0
CTHMM_batch_decode_Etij_for_subjects(0, df_test, response_list, fitted.Q_mat_fit, fitted.π_list_fit, fitted.state_list_fit)

result_df = combine(groupby(df_test, :ID)) do sub_df
        prop1 = sum(skipmissing(sub_df.Sv1 .* sub_df.time_interval)) / sum(skipmissing(sub_df.time_interval))
        prop2 = sum(skipmissing(sub_df.Sv2 .* sub_df.time_interval)) / sum(skipmissing(sub_df.time_interval))
        prop3 = sum(skipmissing(sub_df.Sv3 .* sub_df.time_interval)) / sum(skipmissing(sub_df.time_interval))
        dur = sum(skipmissing(sub_df.time_interval))
        return DataFrame(Prop1 = prop1, Prop2 = prop2, Prop3 = prop3, Duration = dur)
end

avg_trip_duration = 1300
tmp = π_list0' * exp(Q_mat0 * 1)
for t = collect(2:1:avg_trip_duration)
        tmp = tmp + π_list0' * exp(Q_mat0 * t)
end
tmp / avg_trip_duration


sum(skipmissing((df_sim.true_state .== 1) .* df_sim.time_interval)) / sum(skipmissing(df_sim.time_interval))
sum(skipmissing((df_sim.true_state .== 2) .* df_sim.time_interval)) / sum(skipmissing(df_sim.time_interval))
sum(skipmissing((df_sim.true_state .== 3) .* df_sim.time_interval)) / sum(skipmissing(df_sim.time_interval))

sum(df_sim.true_state .== 1) / nrow(df_sim)
sum(df_sim.true_state .== 2) / nrow(df_sim)
sum(df_sim.true_state .== 3) / nrow(df_sim)

exp(Q_mat0 * 10)


# # cmm for parameter initial guess
# kmeans(hcat(df_sim.response1), 1)

# histogram(df_sim.response1)





