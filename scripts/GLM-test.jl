using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, CategoricalArrays, GLM, QuasiGLM, Random, Base.Threads
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
include(srcdir("CTHMM-learn-cov-EM.jl"))
# include(srcdir("CTHMM-build-cov-Q.jl"))   # imported in CTHMM.jl to use @check_args; call CTHMM.build_cov_Q instead

α_true = [0.25 0.5 -0.3;
    0 0 0]

Random.seed!(1234)
subject_df = DataFrame(subject_ID = collect(1:1:50),
                        intercept = 1,
                        covariate1 = rand(Distributions.Uniform(0, 2), 50),
                        covariate2 = rand(Distributions.Normal(1, 1), 50))

# just assume 1 π_list for all drivers for now
π_list_true = [0.7; 0.3]
state_list_true = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-3, 2)]
response_list = ["response1"]

covariate_list = ["intercept", "covariate1", "covariate2"]
Random.seed!(1234)
df_sim = sim_dataset_Qn(α_true, subject_df, covariate_list, π_list_true, state_list_true, 50)


α_init = [0.26 0.7 0.2; 0.0 0.0 0.0]
π_list_init = [0.7; 0.3]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

fitted = CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
    max_iter = 1000, α_max_iter = 5, penalty = false)

# (α_fit = [0.24804284370256433 0.5153400348290043; 0.0 0.0], π_list_fit = [0.6986277718858532, 0.30137222811414677], state_list_fit = Main.CTHMM.NormalExpert{Float64}[Main.CTHMM.NormalExpert{Float64}(-0.004642361128551442, 1.004177340221124) Main.CTHMM.NormalExpert{Float64}(-2.9822755795314464, 2.0143185017758656)], converge = true, iter = 952, ll = -296175.08295187063)

# [0.2556727617381064 0.5172955793133771; 0.0 0.0]

# (α_fit = [0.2390660768692948 0.5453853920616729; 0.0 0.0], π_list_fit = [0.7408893658937321, 0.25911063410626795], state_list_fit = Main.CTHMM.NormalExpert{Float64}[Main.CTHMM.NormalExpert{Float64}(0.004432273383086754, 0.9884524495459106) Main.CTHMM.NormalExpert{Float64}(-2.962825409049089, 2.0180029547281224)], converge = true, iter = 333, ll = -296756.32455303636)


α_true = [0.25; 0]

Random.seed!(1234)
subject_df = DataFrame(subject_ID = 1,
                        intercept = 1)

# just assume 1 π_list for all drivers for now
π_list_true = [0.7; 0.3]
state_list_true = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-3, 2)]
response_list = ["response1"]

covariate_list = ["intercept"]
Random.seed!(1234)
df_sim = sim_dataset_Qn(α_true, subject_df, covariate_list, π_list_true, state_list_true, 1000)


α_init = [0.26; 0.0]
π_list_init = [0.7; 0.3]

state_list_init = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1)]

fitted_cov = CTHMM_learn_cov_EM(df_sim, response_list, subject_df, covariate_list, α_init, π_list_init, state_list_init; 
    max_iter = 1000, α_max_iter = 5, penalty = false)



# # need to reverse the signs for:
# dff = DataFrame(y = subject_df.tau1 ./ subject_df.N12, x = subject_df.covariate1, w = coalesce.(subject_df.N12, 0.0))

y = subject_df.N12 ./ subject_df.tau1
X = subject_df[!, covariate_list]
w = coalesce.(subject_df.tau1, 0.0)

coef(glm(Matrix(X), y, Gamma(), LogLink(), wts = w))

dff = DataFrame(y = subject_df.N12 ./ subject_df.tau1, x = subject_df[!, covariate_list], w = coalesce.(subject_df.tau1, 0.0))

# reversed the responses and weights so can take results directly
dff = DataFrame(y = subject_df.N12 ./ subject_df.tau1, x = subject_df.covariate1, w = coalesce.(subject_df.tau1, 0.0))

coef(glm(@formula(y ~ -1 + x), dff, Gamma(), LogLink(), wts = dff.w))


# write for-loop to change the indices and take coefficient estimates
dff = DataFrame(y = subject_df.N21 ./ subject_df.tau2, x = subject_df.covariate1, w = coalesce.(subject_df.tau2, 0.0))

coef(glm(@formula(y ~ x), dff, Gamma(), LogLink(), wts = dff.w))