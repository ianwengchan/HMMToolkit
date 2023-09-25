using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, QuasiGLM, Random, Base.Threads, Logging, JLD2
using .CTHMM

# include(srcdir("fitting-utils.jl"))
# using .fit_jl

num_subject = 100
num_time_series = 50
Random.seed!(1234)
subject_df = DataFrame(subject_ID = collect(1:1:num_subject),
                        intercept = 1,
                        covariate1 = rand(Distributions.Uniform(0, 2), num_subject),
                        covariate2 = rand(Distributions.Normal(1, 1), num_subject))
covariate_list = ["intercept", "covariate1", "covariate2"]

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


α_init = [0.26 0.7 0.2; 0.1 0.23 -0.4;
    -0.4 0.15 0.24; -0.1 0.1 0.45; 
    0.0 0.0 0.0; 0.0 0.0 0.0]
π_list_init = [0.4; 0.3; 0.3]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

save_name = datadir("cov-3s-3cov")

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