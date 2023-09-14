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
subject_df = DataFrame(subject_ID = collect(1:1:50),
                        intercept = 1,
                        covariate1 = rand(Distributions.Uniform(0, 1), 50))

# just assume 1 π_list for all drivers for now
π_list = [0.7; 0.3]
state_list = [CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(-3, 2)]

covariate_list = ["intercept", "covariate1"]
Random.seed!(1234)
df_sim = sim_dataset_Qn(α, subject_df, covariate_list, π_list, state_list, 30)


α_init = [0.25 0.5;
        -0.25 0.35]

distinct_time_list = CTHMM_precompute_distinct_time_list(df_sim.time_interval)
num_state = size(π_list, 1)
num_dim = size(state_list, 1)
num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series

group_df = groupby(df_sim, :subject_ID)
num_subject = size(group_df, 1)
distinct_time_list = Array{Any}(undef, num_subject) # the distinct_time_list depends on the subject
for n = 1:num_subject
    distinct_time_list[n] = CTHMM_precompute_distinct_time_list(group_df[n].time_interval)
end

for i = 1:num_state
    subject_df[:, string("tau", i)] = missings(Float64, nrow(subject_df))
    for j = 1:num_state
        subject_df[:, string("N", i, j)] = missings(Float64, nrow(subject_df))
    end
end

response_list = ["response1"]

# first, do E-step for all drivers, saving Svi to df; need to compute Etij separately for each subject
ll_em_temp, Etij = CTHMM_batch_decode_Etij_for_cov_subjects(1, df_sim, response_list, subject_df, covariate_list, α, π_list, state_list)

## M-step, with last estimated parameters
## part 1a: learning initial state probabilities π_list
for i in 1:num_state
    π_list[i] = sum(df_sim.start .* df_sim[:, string("Sv", i)])
end
π_list = π_list ./ sum(π_list)

ll_em = CTHMM_batch_decode_for_cov_subjects(1, df_sim, response_list, subject_df, covariate_list, α, π_list, state_list)

s = ll_em - ll_em_temp > 0 ? "+" : "-"
pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
# if (print_steps > 0) & (iter % print_steps == 0)
#     @info(
#         "Iteration $(iter), updating π_list: $(ll_em_temp) ->  $(ll_em), ( $(s) $(pct) % )"
#     )
# end
ll_em_temp = ll_em

α_old = copy(α) .- Inf

α_iter = 0

# while (α_iter < α_max_iter) && (sum((α - α_old) .^ 2) > 1e-10)

    α_iter = α_iter + 1
    α_old = copy(α)

    # have to find Nij_mat and taui_list for each subject
    subject_df = CTHMM_learn_cov_nij_taui(num_state, num_subject, subject_df, covariate_list, distinct_time_list, α, Etij)


# # need to reverse the signs for:
# dff = DataFrame(y = subject_df.tau1 ./ subject_df.N12, x = subject_df.covariate1, w = coalesce.(subject_df.N12, 0.0))

# reversed the responses and weights so can take results directly
dff = DataFrame(y = subject_df.N12 ./ subject_df.tau1, x = subject_df.covariate1, w = coalesce.(subject_df.tau1, 0.0))

coef(glm(@formula(y ~ x), dff, Gamma(), LogLink(), wts = dff.w))


# write for-loop to change the indices and take coefficient estimates
dff = DataFrame(y = subject_df.N21 ./ subject_df.tau2, x = subject_df.covariate1, w = coalesce.(subject_df.tau2, 0.0))

coef(glm(@formula(y ~ x), dff, Gamma(), LogLink(), wts = dff.w))


# α = [0.25 0.5;
#     -0.25 0.35]