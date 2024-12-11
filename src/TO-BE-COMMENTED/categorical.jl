"""
    CategoricalExpert(p)

PDF:

```math
f(x; \\p) = ∏_{i=1}^k p_i^{[x=i]}
```

See also: [Categorical Distribution](https://en.wikipedia.org/wiki/Categorical_distribution) (Wikipedia)

"""
struct CategoricalExpert{T<:AbstractVector{<:Real}} <: RealDiscreteExpert
    p::T
    CategoricalExpert{T}(p::T) where {T<:AbstractVector{<:Real}} = new{T}(p)
end

function CategoricalExpert(p::AbstractVector{<:Real}; check_args=true)
    check_args && @check_args(CategoricalExpert, isprobvec(p))
    return CategoricalExpert{typeof(p)}(p)
end

function CategoricalExpert(k::Integer; check_args=true)
    check_args && @check_args(CategoricalExpert, k >= 1)
    p = fill(1/k, k)
    return CategoricalExpert{typeof(p)}(fill(1/k, k); check_args=false)
end

## Outer constructors
# CategoricalExpert(p::AbstractVector{<:Real}) = CategoricalExpert(promote(p)...)
# CategoricalExpert(k::Integer) = CategoricalExpert(float(p))

## Conversion
function convert(::Type{CategoricalExpert{T}}, p::S) where {T<:AbstractVector{<:Real}, S<:AbstractVector{<:Real}}
    return CategoricalExpert(T(p))
end
function convert(::Type{CategoricalExpert{T}}, d::CategoricalExpert{S}) where {T<:AbstractVector{<:Real}, S<:AbstractVector{<:Real}}
    return CategoricalExpert(T(d.p); check_args=false)
end
copy(d::CategoricalExpert) = CategoricalExpert(d.p; check_args=false)

## Loglikelihood of Expert
function logpdf(d::CategoricalExpert, x...)
    return Distributions.logpdf.(Distributions.Categorical(d.p), x...)
end
pdf(d::CategoricalExpert, x...) = Distributions.pdf.(Distributions.Categorical(d.p), x...)
function logcdf(d::CategoricalExpert, x...)
    return Distributions.logcdf.(Distributions.Categorical(d.p), x...)
end
cdf(d::CategoricalExpert, x...) = Distributions.cdf.(Distributions.Categorical(d.p), x...)

## expert_ll, etc
expert_ll_exact(d::CategoricalExpert, x::Real) = HMMToolkit.logpdf(d, x)
function expert_ll(d::CategoricalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end
# function expert_tn(d::NormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
#     expert_tn = if (tl == tu)
#         logpdf.(d, tl)
#     else
#         logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
#     end
#     expert_tn = (tu == 0.0) ? -Inf : expert_tn
#     return expert_tn
# end
# function expert_tn_bar(d::NormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
#     expert_tn_bar = if (tl == tu)
#         0.0
#     else
#         log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))
#     end
#     return expert_tn_bar
# end

exposurize_expert(d::CategoricalExpert; exposure=1) = d

## Parameters
ncategories(d::CategoricalExpert) = Distributions.ncategories(Distributions.Categorical(d.p))
params(d::CategoricalExpert) = (d.p)
function params_init(y, d::CategoricalExpert)
    k = ncategories(d)
    y = collect(skipmissing(y))
    # pos_idx = (y .> 0.0)  # Normal distribution takes negative values as well
    p_init = [count(==(i), y) / length(y) for i in 1:k]
    p_init = any(isnan, p_init) ? (fill(1/k), k) : p_init
    return CategoricalExpert(p_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::CategoricalExpert)
    # p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    # return max(
    #     abs(p_zero - 0.0),
    #     (1 - 0.0) *
    #     HypothesisTests.ksstats(y[y .> 0.0], Distributions.Normal(d.μ, d.σ))[2],
    # )
    return HypothesisTests.ksstats(y, Distributions.Categoricald.p)[2]
end

## Simulation
sim_expert(d::CategoricalExpert) = Distributions.rand(Distributions.Categorical(d.p), 1)[1]

## penalty
penalty_init(d::CategoricalExpert) = [2.0]
no_penalty_init(d::CategoricalExpert) = [1.0]
penalize(d::CategoricalExpert, p) = (p[1] - 1)*sum(log.(d.p))

## statistics
mean(d::CategoricalExpert) = mean(Distributions.Categorical(d.p))
var(d::CategoricalExpert) = var(Distributions.Categorical(d.p))
quantile(d::CategoricalExpert, p) = quantile(Distributions.Categorical(d.p), p)
# function lev(d::NormalExpert, u)
#     if isinf(u)
#         return mean(d)
#     else
#         return exp(d.μ + 0.5 * d.σ^2) * cdf.(Normal(d.μ + d.σ^2, d.σ), log(u)) +
#                u * (1 - cdf.(Normal(d.μ, d.σ), log(u)))
#     end
# end
# excess(d::NormalExpert, u) = mean(d) - lev(d, u)

# ## Misc functions for E-Step
# function _diff_dens_series(d::NormalExpert, yl, yu)
#     return exp(-0.5 * (log(yl) - d.μ)^2 / (d.σ^2)) - exp(-0.5 * (log(yu) - d.μ)^2 / (d.σ^2))
# end

# function _diff_dist_series(d::NormalExpert, yl, yu)
#     return (0.5 + 0.5 * erf((log(yu) - d.μ) / (sqrt2 * d.σ))) -
#            (0.5 + 0.5 * erf((log(yl) - d.μ) / (sqrt2 * d.σ)))
# end

# function _int_obs_logY(d::NormalExpert, yl, yu, expert_ll)
#     if yl == yu
#         return log(yl)
#     else
#         return exp(-expert_ll) * (
#             d.σ * invsqrt2π * _diff_dens_series(d, yl, yu) +
#             d.μ * _diff_dist_series(d, yl, yu)
#         )
#     end
# end

# function _int_lat_logY(d::NormalExpert, tl, tu, expert_tn_bar)
#     return exp(-expert_tn_bar) * (
#         d.μ - (
#             d.σ * invsqrt2π * _diff_dens_series(d, tl, tu) +
#             d.μ * _diff_dist_series(d, tl, tu)
#         )
#     )
# end

# function _zdensz_series(z)
#     return (isinf(z) ? 0.0 : z * exp(-0.5 * z^2))
# end

# function _diff_zdensz_series(d::NormalExpert, yl, yu)
#     return _zdensz_series((log(yl) - d.μ) / d.σ) - _zdensz_series((log(yu) - d.μ) / d.σ)
# end

# function _int_obs_logY_sq(d::NormalExpert, yl, yu, expert_ll)
#     if yl == yu
#         return (log(yl))^2
#     else
#         return exp(-expert_ll) * (
#             ((d.σ)^2 + (d.μ)^2) * _diff_dist_series(d, yl, yu) +
#             2.0 * d.σ * d.μ * invsqrt2π * _diff_dens_series(d, yl, yu) +
#             (d.σ)^2 * invsqrt2π * _diff_zdensz_series(d, yl, yu)
#         )
#     end
# end

# function _int_lat_logY_sq(d::NormalExpert, tl, tu, expert_tn_bar)
#     return exp(-expert_tn_bar) * (
#         ((d.σ)^2 + (d.μ)^2) - (
#             ((d.σ)^2 + (d.μ)^2) * _diff_dist_series(d, tl, tu) +
#             2.0 * d.σ * d.μ * invsqrt2π * _diff_dens_series(d, tl, tu) +
#             (d.σ)^2 * invsqrt2π * _diff_zdensz_series(d, tl, tu)
#         )
#     )
# end

## EM: M-Step
# function EM_M_expert(d::NormalExpert,
#     tl, yl, yu, tu,
#     exposure,
#     z_e_obs, z_e_lat, k_e;
#     penalty=true, pen_params_jk=[1.0 1.0])

#     expert_ll_pos = expert_ll.(d, tl, yl, yu, tu)
#     expert_tn_bar_pos = expert_tn_bar.(d, tl, yl, yu, tu)

#     # Further E-Step
#     logY_e_obs = vec(_int_obs_logY.(d, yl, yu, expert_ll_pos))
#     logY_e_lat = vec(_int_lat_logY.(d, tl, tu, expert_tn_bar_pos))
#     nan2num(logY_e_lat, 0.0) # get rid of NaN

#     logY_sq_e_obs = vec(_int_obs_logY_sq.(d, yl, yu, expert_ll_pos))
#     logY_sq_e_lat = vec(_int_lat_logY_sq.(d, tl, tu, expert_tn_bar_pos))
#     nan2num(logY_sq_e_lat, 0.0) # get rid of NaN

#     # Update parameters
#     pos_idx = (yu .!= 0.0)
#     term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
#     term_zkz_logY =
#         (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) .+
#         (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])
#     term_zkz_logY_sq =
#         (z_e_obs[pos_idx] .* logY_sq_e_obs[pos_idx]) .+
#         (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_sq_e_lat[pos_idx])

#     μ_new = sum(term_zkz_logY)[1] / sum(term_zkz)[1]

#     demominator = penalty ? (sum(term_zkz)[1] + (pen_params_jk[2] - 1)) : sum(term_zkz)[1]
#     numerator = if penalty
#         (
#             sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
#             (μ_new)^2 * sum(term_zkz)[1] + (pen_params_jk[1] - 1)
#         )
#     else
#         (
#             sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
#             (μ_new)^2 * sum(term_zkz)[1]
#         )
#     end
#     tmp = numerator / demominator
#     σ_new = sqrt(maximum([0.0, tmp]))

#     return NormalExpert(μ_new, σ_new)
# end

## EM: M-Step, exact observations
function EM_M_expert_exact(
    d::CategoricalExpert,
    ye, # exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[1.0])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    k = ncategories(d)
    # Update parameters
    term_zkz_Y_ind = [sum((ye .== i) .* z_e_obs) for i in 1:k]

    numerator = penalty ? (term_zkz_Y_ind .+ (pen_params_jk[1] - 1)) : term_zkz_Y_ind
    p_new = numerator / sum(numerator)

    p_new = any(isnan, p_new) ? (fill(1/k), k) : p_new

    return CategoricalExpert(p_new)
end