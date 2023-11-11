using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, Random, Base.Threads, Logging, JLD2, StatsPlots
using .CTHMM

# include(srcdir("fitting-utils.jl"))
# using .fit_jl

subject_df_o = load(datadir("Splend_Test/subject_df.jld2"), "subject_df")
df_longer = load(datadir("Splend_Test/df_longer.jld2"), "df_longer")

ux = unique(subject_df_o.Make)
transform!(subject_df_o, @. :Make => ByRow(isequal(ux)) .=> Symbol(:Make_, ux))
ux = unique(subject_df_o.VehicleType)
transform!(subject_df_o, @. :VehicleType => ByRow(isequal(ux)) .=> Symbol(:VehicleType_, ux))

subject_df = subject_df_o[!, [:SubjectId, :Make_Hyundai, :Make_Mitsubishi, :Year, :VehicleType_MPV]]
subject_df.Intercept .= 1

num_subject = 100
# num_time_series = 50
covariate_list = ["Intercept", "Make_Hyundai", "Make_Mitsubishi", "Year", "VehicleType_MPV"]

# just assume 1 π_list for all drivers for now; 2 states
α_init = [0.5 0.5 0.5 0.5 0.5;
        0 0 0 0 0]
π_list_init = [0.6; 0.4]
state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0,3)]
response_list = ["acceleration"]

density(skipmissing(df_longer.radian))  # 6 modes
density(skipmissing(df_longer.delta_radian))    # 4 modes: -1.5, 0, 1.5, 3.14??
density(filter(x -> x <= 60, skipmissing(df_longer.time_interval)))     # 5 sec is the mode
density(skipmissing(df_longer.Speed))   # 3/4 modes: 0, 30, 40, 120
density(skipmissing(df_longer.acceleration))    # 0 is the mode
density(filter(x -> (x <= 3 || x >= -3), skipmissing(df_longer.acceleration)))    # 0 is the mode

CTHMM_learn_cov_EM(df_longer, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
    max_iter = 2000, α_max_iter = 5, penalty = false)



# just assume 1 π_list for all drivers for now
α_true = [0.25 0.5 -0.3;
    0.1 0.24 -0.5;
    -0.5 0.1 0.25;
    -0.1 0.2 0.5;
    0 0 0;
    0 0 0]
π_list_true = [0.5; 0.35; 0.15]
state_list_true = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-3, 2) CTHMM.NormalExpert(3,3)]
response_list = ["response1"]

Random.seed!(1234)
df_sim = sim_dataset_Qn(α_true, subject_df, covariate_list, π_list_true, state_list_true, num_time_series)

α_init = [0.2601 0.7 0.2; 0.1 0.23 -0.4;
    -0.4 0.15 0.24; -0.1 0.1 0.45; 
    0.0 0.0 0.0; 0.0 0.0 0.0]
π_list_init = [0.4; 0.3; 0.3]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

save_name = datadir("cov-3s-3cov-gamma")

# CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
#     max_iter = 2000, α_max_iter = 5, penalty = false)

function fit_CTHMM_covariate_model(df, response_list, subject_df, covariate_list, α, π_list, state_list, save_name; kwargs...)
    # log_name = "model"*"$(i)"*".txt"
    io = open("$(save_name)" * ".txt", "w+")
    logger = ConsoleLogger(io)
    with_logger(logger) do
        @info(now())
        @info("Fitting $(save_name) started.")
        try
            result = CTHMM_learn_cov_EM(df, response_list, subject_df, covariate_list, α, π_list, state_list; kwargs...)
            @save "$(save_name)" * ".JLD2" result
        catch
        end
        @info(now())
        @info("Fitting $(save_name) ended.")
    end
    return close(io)
end

fit_CTHMM_covariate_model(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init, save_name; 
    max_iter = 2000, α_max_iter = 5, penalty = false)

# CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
#     max_iter = 2000, α_max_iter = 5, penalty = false)