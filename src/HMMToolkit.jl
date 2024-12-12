module HMMToolkit

import Base: size, length, convert, show, getindex, rand, vec, inv, expm1, abs, log1p
import Base: isnan, isinf
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base: convert, copy, findfirst, summary
import Base.Math: @horner
import Base: π
import Base.Threads: @threads, nthreads, threadid

using StatsBase
import StatsBase: weights

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp
import StatsFuns: sqrt2, invsqrt2π

using Statistics
import Statistics: quantile, mean, var, median

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf, quantile
import Distributions: rand, AbstractRNG
import Distributions: mean, var, skewness, kurtosis
import Distributions:
    UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: @distr_support, RecursiveProbabilityEvaluator
import Distributions: Bernoulli, Multinomial
import Distributions: Binomial, Poisson
import Distributions: Gamma, InverseGaussian, LogNormal, Normal, Weibull
import Distributions: Laplace, VonMises

using InvertedIndices
import InvertedIndices: Not

using LinearAlgebra
import LinearAlgebra: I, Cholesky

using SpecialFunctions
import SpecialFunctions: erf, loggamma, gamma_inc, gamma, beta_inc

using QuadGK
import QuadGK: quadgk

using Optim
import Optim: optimize, minimizer

using Clustering
import Clustering: kmeans, assignments, counts

using HypothesisTests
import HypothesisTests: ExactOneSampleKSTest, pvalue, ksstats

using Logging

using Roots
import Roots: find_zero, Order2

using Bessels
import Bessels: besseli0, besseli1

using DataFrames
import DataFrames: DataFrame

using GLM

using HypothesisTests

using ShiftedArrays

using StatsPlots

using Distances, DynamicAxisWarping

export
    ##### CTHMM
    ## init
    cluster_responses,
    CTHMM_cmm_init_exact,
    CTHMM_cmm_init,
    ## experts
    params,
    p_zero,
    params_init,
    GammaExpert, ZIGammaExpert,
    LogNormalExpert, ZILogNormalExpert,
    LaplaceExpert,
    NormalExpert,
    VonMisesExpert,
    # precompute
    CTHMM_precompute_distinct_time_list,
    CTHMM_precompute_distinct_time_Pt_list,
    CTHMM_precompute_distinct_time_log_Pt_list,
    CTHMM_precompute_batch_data_emission_prob,
    CTHMM_precompute_batch_data_emission_prob_separate,
    CTHMM_precompute_batch_data_emission_cdf_separate,
    # decode
    CTHMM_decode_forward_backward,
    CTHMM_likelihood_forward,
    CTHMM_decode_viterbi,
    CTHMM_batch_decode_Etij_for_subjects,
    CTHMM_batch_decode_Etij_and_append_Svi_for_subjects,
    CTHMM_batch_decode_for_subjects_list,
    CTHMM_batch_decode_for_subjects,
    # learning
    penalty_params,
    CTHMM_learn_nij_taui,
    CTHMM_learn_cov_nij_taui,
    CTHMM_learn_update_Q_mat,
    CTHMM_learn_EM,
    CTHMM_learn_EM_Q_only,
    CTHMM_learn_EM_state_only,
    # simulation
    CTHMM_sim_time_series,
    CTHMM_sim_dataset,
    # analysis
    CTHMM_ordinary_pseudo_residuals,
    CTHMM_forecast_pseudo_residuals,
    CTHMM_pseudo_residuals,
    CTHMM_batch_pseudo_residuals,
    CTHMM_anomaly_indices,
    CTHMM_batch_anomaly_indices,

    ##### DTHMM
    # init
    DTHMM_cmm_init_exact,
    DTHMM_cmm_init,
    # decode
    DTHMM_decode_forward_backward,
    DTHMM_likelihood_forward,
    DTHMM_decode_viterbi,
    DTHMM_batch_decode_Eij_for_subjects,
    DTHMM_batch_decode_Eij_and_append_Svi_for_subjects,
    DTHMM_batch_decode_for_subjects_list,
    DTHMM_batch_decode_for_subjects,
    # learning
    DTHMM_learn_update_P_mat,
    DTHMM_learn_EM,
    # simulation
    DTHMM_sim_time_series,
    DTHMM_sim_dataset,
    # analysis
    DTHMM_ordinary_pseudo_residuals,
    DTHMM_forecast_pseudo_residuals,
    DTHMM_batch_pseudo_residuals,
    DTHMM_batch_anomaly_indices


### source files

include("utils.jl")
# include("modelstruct.jl")

include("expert.jl")
include("penalty.jl")

include("paramsinit.jl") #commented
include("em.jl")

# Continuous-time HMM
include("CTHMM-precompute-distinct-time.jl") #commented
include("CTHMM-precompute.jl") #commented
include("CTHMM-decode-forward-backward.jl") #commented
include("CTHMM-decode-viterbi.jl") #commented
include("CTHMM-batch-decode.jl") #commented
include("CTHMM-learn-nij-taui.jl") #commented
include("CTHMM-learn-Q.jl") #commented
include("CTHMM-learn-EM.jl") #commented
include("CTHMM-simulation.jl") #commented
include("CTHMM-pseudo-residuals.jl") #commented

# Discrete-time HMM
include("DTHMM-decode-forward-backward.jl") #commented
include("DTHMM-decode-viterbi.jl") #commented
include("DTHMM-batch-decode.jl") #commented
include("DTHMM-learn-P.jl") #commented
include("DTHMM-learn-EM.jl") #commented
include("DTHMM-simulation.jl") #commented
include("DTHMM-pseudo-residuals.jl") #commented

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for both CTHMM and DTHMM fitting and anomaly detection.
"""
HMMToolkit

end # module