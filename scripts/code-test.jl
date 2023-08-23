using DrWatson
@quickactivate "FitHMM-jl"

include(srcdir("CTHMM.jl"))
using CSV, DataFrames, Dates, Statistics, LinearAlgebra, Distributions, JLD2
using .CTHMM

# CTHMM-precompute-distinct-time.jl - tested
# CTHMM-precompute.jl - partially tested
# CTHMM-decode-forward-backward.jl - tested
# CTHMM-decode-viterbi.jl - tested
# CTHMM-batch-decode.jl - tested
# CTHMM-learn-nij-taui.jl - tested
# CTHMM-learn-Q.jl - tested
# CTHMM-

include(srcdir("CTHMM-precompute-distinct-time.jl"))
include(srcdir("CTHMM-precompute.jl"))
include(srcdir("CTHMM-decode-forward-backward.jl"))
include(srcdir("CTHMM-decode-viterbi.jl"))
include(srcdir("CTHMM-batch-decode.jl"))
include(srcdir("CTHMM-learn-nij-taui.jl"))
include(srcdir("CTHMM-learn-Q.jl"))

df_longer = load(datadir("df_longer.jld2"), "df_longer")

# test with a CTHMM with 4 states, initiate Q
Q_mat0 = [-1.5 0.5 0.5 0.5;
        0.1 -0.3 0.1 0.1;
        0.2 0.2 -0.6 0.2;
        0.5 0.5 0.5 -1.5]

π_list0 = [0.5; 0.25; 0.15; 0.1]

response_list = ["delta_radian", "acceleration"]

df = Base.copy(df_longer)
Q_mat_init = Base.copy(Q_mat0)
π_list_init = Base.copy(π_list0)

state_list_init = [CTHMM.NormalExpert(0, 0.5) CTHMM.NormalExpert(0, 1) CTHMM.NormalExpert(0, 1.5) CTHMM.NormalExpert(0, 2);
                CTHMM.NormalExpert(1, 0.5) CTHMM.NormalExpert(1, 1) CTHMM.NormalExpert(1, 1.5) CTHMM.NormalExpert(1, 2)]

soft_decode = 1
ϵ = 1e-03
max_iter = 20
print_steps = 1

## precomputation before iteration
# only once in the whole EM:
distinct_time_list = CTHMM_precompute_distinct_time_list(df.time_interval)
num_state = size(Q_mat_init, 1)
num_dim = size(state_list_init, 1)
num_distinct_time = size(distinct_time_list, 1) # assume one Q for all time series

# start EM iteration
Q_mat = Base.copy(Q_mat_init)
π_list = Base.copy(π_list_init)
state_list = Base.copy(state_list_init)
ll_em_old = -Inf
ll_em = CTHMM_batch_decode_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)
iter = 0


ll_em, Etij = CTHMM_batch_decode_Etij_for_subjects(soft_decode, df, response_list, Q_mat, π_list, state_list)

ye = df.delta_radian
z_e_obs = df.Sv1



# Step 3: Calculate the weighted median
function weighted_median(obs, weights)
        sorted_data = sort(DataFrame(obs=obs, weights=weights), :obs)
        total_weight = sum(weights)

        cum_weights = cumsum(sorted_data.weights)
        midpoint = total_weight / 2.0

        idx = findfirst(cum_weights .>= midpoint)
        if isodd(length(weights))
                return sorted_data[idx, :obs]
        else
                prev_idx = idx - 1
                return (sorted_data[prev_idx, :obs] + sorted_data[idx, :obs]) / 2.0
        end
end
    
obs = data[:, :Observations]
weights = data[:, :Weights]

weighted_median(obs, weights)

