using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, QuasiGLM, Random
using .CTHMM

# data = DataFrame(y = rand(100).* 3, x = categorical(repeat([1,2,3,4], 25)))
# glm(@formula(y ~ x), data, Poisson())

include(srcdir("CTHMM-precompute-distinct-time.jl"))
include(srcdir("CTHMM-precompute.jl"))
include(srcdir("CTHMM-decode-forward-backward.jl"))
include(srcdir("CTHMM-decode-viterbi.jl"))
include(srcdir("CTHMM-batch-decode.jl"))
include(srcdir("CTHMM-learn-nij-taui.jl"))
include(srcdir("CTHMM-learn-Q.jl"))
include(srcdir("CTHMM-learn-EM.jl"))
include(srcdir("CTHMM-simulation.jl"))
# include(srcdir("CTHMM-build-cov-Q.jl"))   # imported in CTHMM.jl to use @check_args; call CTHMM.build_cov_Q instead

α = [0.25 0.5;
    -0.25 0.35]

Random.seed!(1234)
driver_df = DataFrame(driver_ID = collect(1:1:50),
                        intercept = 1,
                        covariate1 = rand(Distributions.Uniform(0, 1), 50))

# just assume 1 π_list for all drivers for now
π_list = [0.7; 0.3]
state_list = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-3, 2)]

covariate_list = ["intercept", "covariate1"]
Random.seed!(1234)
df_sim = sim_dataset_Qn(α, driver_df, covariate_list, π_list, state_list, 30)