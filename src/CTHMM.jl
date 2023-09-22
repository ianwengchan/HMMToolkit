module CTHMM

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

# export
#     ## generic types
#     # convert,
#     # copy,
#     # ZeroInflation,
#     # ExpertSupport,
#     # AnyExpert,
#     # DiscreteExpert,
#     # ContinuousExpert,
#     # RealDiscreteExpert,
#     # RealContinuousExpert,
#     # NonNegDiscreteExpert,
#     # NonNegContinuousExpert,
#     # ZIDiscreteExpert,
#     # ZIContinuousExpert,
#     # NonZIDiscreteExpert,
#     # NonZIContinuousExpert,

#     ## model related
#     summary,
#     # LRMoEModel,
#     # LRMoESTD,
#     # LRMoEFittingResult,
#     # LRMoESTDFit,

#     ## loglikelihood functions
#     pdf, logpdf,
#     cdf, logcdf,
#     rowlogsumexp,
#     expert_ll,
#     expert_tn,
#     expert_tn_bar,
#     # expert_ll_list,
#     # expert_tn_list,
#     # expert_tn_bar_list, loglik_aggre_dim,
#     # loglik_aggre_gate_dim,
#     # loglik_aggre_gate_dim_comp,
#     # loglik_np,
#     # loglik_exact, penalty_α,
#     # penalty_params,
#     # penalize,

#     ## EM related
#     # nan2num,
#     # EM_E_z_obs,
#     # EM_E_z_lat,
#     # EM_E_k,
#     # EM_M_α,
#     # EM_M_dQdα,
#     # EM_M_dQ2dα2,
#     # EM_E_z_zero_obs_update,
#     # EM_E_z_zero_lat_update,
#     # EM_E_z_zero_obs,
#     # EM_E_z_zero_lat,

#     ## init
#     cmm_init,
#     cmm_init_exact,

#     ## gating
#     LogitGating,

#     ## experts
#     Burr,
#     GammaCount, params,
#     p_zero,
#     params_init,
#     exposurize_expert,
#     exposurize_model, BurrExpert, ZIBurrExpert,
#     GammaExpert, ZIGammaExpert,
#     InverseGaussianExpert, ZIInverseGaussianExpert,
#     LogNormalExpert, ZILogNormalExpert,
#     WeibullExpert, ZIWeibullExpert, BinomialExpert, ZIBinomialExpert,
#     GammaCountExpert, ZIGammaCountExpert,
#     NegativeBinomialExpert, ZINegativeBinomialExpert,
#     PoissonExpert, ZIPoissonExpert,

#     ## fitting
#     fit_main,
#     fit_exact,
#     fit_LRMoE,

#     ## simulation
#     sim_expert,
#     sim_logit_gating,
#     sim_dataset,

#     # prediction
#     mean, var, quantile,
#     predict_class_prior,
#     predict_class_posterior,
#     predict_mean_prior,
#     predict_mean_posterior,
#     predict_var_prior,
#     predict_var_posterior,
#     predict_limit_prior,
#     predict_limit_posterior,
#     predict_excess_prior,
#     predict_excess_posterior,
#     predict_VaRCTE_prior,
#     predict_VaRCTE_posterior

export
    # precompute
    CTHMM_precompute_distinct_time_list,
    CTHMM_precompute_distinct_time_Pt_list,
    CTHMM_precompute_distinct_time_log_Pt_list,
    CTHMM_precompute_batch_data_emission_prob,
    CTHMM_precompute_batch_data_emission_log_prob,
    # decode
    CTHMM_decode_forward_backward,
    CTHMM_likelihood_forward,
    CTHMM_likelihood_true,
    CTHMM_decode_viterbi,
    CTHMM_batch_decode_Etij_for_subjects,
    CTHMM_batch_decode_Etij_and_append_Svi_for_subjects,
    CTHMM_batch_decode_for_subjects,
    CTHMM_batch_decode_Etij_for_cov_subjects,
    CTHMM_batch_decode_Etij_and_append_Svi_for_cov_subjects,
    CTHMM_batch_decode_for_cov_subjects,
    # learning
    CTHMM_learn_nij_taui,
    CTHMM_learn_cov_nij_taui,
    CTHMM_learn_update_Q_mat,
    CTHMM_learn_EM,
    CTHMM_learn_cov_EM,
    # simulation
    sim_time_series,
    sim_dataset,
    sim_dataset_Qn


### source files

include("utils.jl")
# include("AICBIC.jl")
# include("modelstruct.jl")

# include("gating.jl")
include("expert.jl")
# include("loglik.jl")
# include("penalty.jl")

# include("paramsinit.jl")
# include("fit.jl")
include("em.jl")
include("CTHMM-build-cov-Q.jl")

# include("CTHMM-precompute-distinct-time.jl")
# include("CTHMM-precompute.jl")
# include("CTHMM-decode-forward-backward.jl")
# include("CTHMM-decode-viterbi.jl")
# include("CTHMM-batch-decode.jl")
# include("CTHMM-learn-nij-taui.jl")
# include("CTHMM-learn-Q.jl")
# include("CTHMM-learn-EM.jl")
# include("CTHMM-simulation.jl")
# include("CTHMM-learn-cov-EM.jl")

# include("simulation.jl")
# include("predict.jl")

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for the CTHMM model.
"""
CTHMM

end # module