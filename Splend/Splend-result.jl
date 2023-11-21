using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, Random, Base.Threads, Logging, JLD2, StatsPlots, StatsBase
using Plots
using .CTHMM

# include(srcdir("fitting-utils.jl"))
# using .fit_jl

subject_df_o = load(datadir("Splend_Test/subject_df.jld2"), "subject_df")
df_longer = load(datadir("Splend_Test/df_longer.jld2"), "df_longer")

group_df = groupby(df_longer, [:SubjectId, :ID])
trip = group_df[1]
plot(trip.time_since, trip.Speed, label = "Speed")
# savefig(plotsdir("speed.png"))

plot(trip.time_since, [trip.radian trip.delta_radian], label = ["Angle" "Angle Change"])
plot(trip.time_since, trip.delta_radian, label = "Angle (radian)")
# savefig(plotsdir("angle.png"))

density(skipmissing(df_longer.radian), label = "Angle")  # 6 modes
density!(skipmissing(df_longer.delta_radian), label = "Angle Change")    # 4 modes: -1.5, 0, 1.5, 3.14??
# savefig(plotsdir("angle-density.png"))


density(filter(x -> x <= 60, skipmissing(df_longer.time_interval)))     # 5 sec is the mode
density(skipmissing(df_longer.Speed), label = "Speed")   # 3/4 modes: 0, 30, 40, 120
# savefig(plotsdir("speed-density.png"))
density(skipmissing(df_longer.acceleration), label = "Acceleration")    # 0 is the mode
# savefig(plotsdir("acceleration-density.png"))
density(filter(x -> (x <= 3 || x >= -3), skipmissing(df_longer.acceleration)))    # 0 is the mode


ux = unique(subject_df_o.Make)
transform!(subject_df_o, @. :Make => ByRow(isequal(ux)) .=> Symbol(:Make_, ux))
ux = unique(subject_df_o.VehicleType)
transform!(subject_df_o, @. :VehicleType => ByRow(isequal(ux)) .=> Symbol(:VehicleType_, ux))

subject_df = subject_df_o[!, [:SubjectId, :Make_Hyundai, :Make_Mitsubishi, :VehicleType_MPV]]
subject_df.Year = (subject_df_o.Year .- mean(subject_df_o.Year)) ./ std(subject_df_o.Year)
subject_df.Intercept .= 1

num_subject = 100
# num_time_series = 50
covariate_list = ["Intercept", "Make_Hyundai", "Make_Mitsubishi", "Year", "VehicleType_MPV"]
response_list = ["acceleration"]


# result = load(datadir("Splend_Result/s3-acceleration.jld"), "fitted")
# @save datadir("Splend_Result/s3-acceleration.JLD2") result

result = load(datadir("Splend_Result/s4-delta_radian.jld2"), "result")
result = load(datadir("Splend_Result/s3-acceleration.jld2"), "result")

subject_df[:, covariate_list]

result.α_fit
result.π_list_fit
result.state_list_fit

result.state_list_fit[1,:]

Q1 = CTHMM.build_cov_Q(3, result.α_fit, hcat(subject_df[1, covariate_list]...))

P1 = exp(Q1)
sum(P1 * P1, dims = 1) .*100 ./3

P = exp(Q1*1000)
sum(P, dims = 1) .*100 ./3

Svi_full, cur_all_subject_prob, Etij_list = CTHMM.CTHMM_batch_decode_Etij_for_cov_subjects(1, df_longer, response_list, subject_df, covariate_list, 
result.α_fit, result.π_list_fit, result.state_list_fit)

Svi_full
getindex.(findmax(Svi_full, dims = 2)[2], [2])