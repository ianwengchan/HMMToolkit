using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, QuasiGLM, Random, Base.Threads, Logging, JLD2
using .CTHMM

# include(srcdir("fitting-utils.jl"))
# using .fit_jl

df_longer = load(datadir("df_longer_1017.jld2"), "df_longer")

dff = df_longer[df_longer.HardwareId .== 552643775, :]

g = 1

# idx = unique(df_longer[:, [:HardwareId, :ID]])
# idx.idx = 1:nrow(idx)
# df_longer = innerjoin(df_longer, idx, on = [:HardwareId, :ID])
# df_longer.idxx = df_longer.ID
# df_longer.ID = df_longer.idx

# num_time_series = nrow(idx)

# just assume 1 π_list for all drivers for now
response_list = ["Speed", "time_interval", "delta_radian", "acceleration"]
π_list_init = [0.4; 0.3; 0.3]
Q_mat_init = [-1 0.5 0.5;
    1 -2 1;
    0.25 0.25 -0.5]

state_list_init = [CTHMM.ZIGammaExpert(0.01, 2, 20) CTHMM.ZIGammaExpert(0.01, 2, 20) CTHMM.ZIGammaExpert(0.01, 2, 20);
                    CTHMM.GammaExpert(1, 25) CTHMM.GammaExpert(1, 25) CTHMM.GammaExpert(1, 25);
                    CTHMM.VonMisesExpert(0, 1) CTHMM.VonMisesExpert(0, 1) CTHMM.VonMisesExpert(0, 1);
                    CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(0, 2) CTHMM.NormalExpert(0, 2)]

save_name = datadir("sample-no-covariate-1017")

# CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
#     max_iter = 2000, α_max_iter = 5, penalty = false)

# function fit_CTHMM_covariate_model(df, response_list, subject_df, covariate_list, α, π_list, state_list, save_name; kwargs...)
#     # log_name = "model"*"$(i)"*".txt"
#     io = open("$(save_name)" * ".txt", "w+")
#     logger = ConsoleLogger(io)
#     with_logger(logger) do
#         @info(now())
#         @info("Fitting $(save_name) started.")
#         try
#             result = CTHMM_learn_cov_EM(df, response_list, subject_df, covariate_list, α, π_list, state_list; kwargs...)
#             @save "$(save_name)" * ".JLD2" result
#         catch
#         end
#         @info(now())
#         @info("Fitting $(save_name) ended.")
#     end
#     return close(io)
# end

function fit_CTHMM_model(df, response_list, π_list, state_list, save_name; kwargs...)
    # log_name = "model"*"$(i)"*".txt"
    io = open("$(save_name)" * ".txt", "w+")
    logger = ConsoleLogger(io)
    with_logger(logger) do
        @info(now())
        @info("Fitting $(save_name) started.")
        try
            result = CTHMM_learn_EM(df, response_list, π_list, state_list; kwargs...)
            @save "$(save_name)" * ".JLD2" result
        catch
        end
        @info(now())
        @info("Fitting $(save_name) ended.")
    end
    return close(io)
end

fit_CTHMM_model(df_longer, response_list, π_list_init, state_list_init, save_name; 
    max_iter = 2000, penalty = false)

# CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
#     max_iter = 2000, α_max_iter = 5, penalty = false)

result = CTHMM_learn_EM(dff, response_list, Q_mat_init, π_list_init, state_list_init; max_iter = 2000, Q_max_iter = 5, penalty = false)