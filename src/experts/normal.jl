"""
    NormalExpert(μ, σ)

PDF:

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\x - \\mu)^2}{2 \\sigma^2} \\right)
```

See also: [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution) (Wikipedia)

"""
struct NormalExpert{T<:Real} <: NonZIContinuousExpert
    μ::T
    σ::T
    NormalExpert{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function NormalExpert(μ::T, σ::T; check_args=true) where {T<:Real}
    check_args && @check_args(NormalExpert, σ >= zero(σ))
    return NormalExpert{T}(μ, σ)
end

## Outer constructors
NormalExpert(μ::Real, σ::Real) = NormalExpert(promote(μ, σ)...)
NormalExpert(μ::Integer, σ::Integer) = NormalExpert(float(μ), float(σ))
NormalExpert() = NormalExpert(0.0, 1.0)

## Conversion
function convert(::Type{NormalExpert{T}}, μ::S, σ::S) where {T<:Real, S<:Real}
    return NormalExpert(T(μ), T(σ))
end
function convert(::Type{NormalExpert{T}}, d::NormalExpert{S}) where {T<:Real, S<:Real}
    return NormalExpert(T(d.μ), T(d.σ); check_args=false)
end
copy(d::NormalExpert) = NormalExpert(d.μ, d.σ; check_args=false)

## Loglikelihood of Expert
function logpdf(d::NormalExpert, x...)
    return Distributions.logpdf.(Distributions.Normal(d.μ, d.σ), x...)
end
pdf(d::NormalExpert, x...) = Distributions.pdf.(Distributions.Normal(d.μ, d.σ), x...)
function logcdf(d::NormalExpert, x...)
    return Distributions.logcdf.(Distributions.Normal(d.μ, d.σ), x...)
end
cdf(d::NormalExpert, x...) = Distributions.cdf.(Distributions.Normal(d.μ, d.σ), x...)

## expert_ll, etc
expert_ll_exact(d::NormalExpert, x::Real) = CTHMM.logpdf(d, x)
function expert_ll(d::NormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
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

exposurize_expert(d::NormalExpert; exposure=1) = d

## Parameters
params(d::NormalExpert) = (d.μ, d.σ)
function params_init(y, d::NormalExpert)
    # pos_idx = (y .> 0.0)  # Normal distribution takes negative values as well
    μ_init, σ_init = mean(y), sqrt(var(y))
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    σ_init = isnan(σ_init) ? 1.0 : σ_init
    return NormalExpert(μ_init, σ_init)
end

# ## KS stats for parameter initialization
# function ks_distance(y, d::NormalExpert)
#     p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
#     return max(
#         abs(p_zero - 0.0),
#         (1 - 0.0) *
#         HypothesisTests.ksstats(y[y .> 0.0], Distributions.Normal(d.μ, d.σ))[2],
#     )
# end

# ## Simululation
# sim_expert(d::NormalExpert) = Distributions.rand(Distributions.Normal(d.μ, d.σ), 1)[1]

# ## penalty
# penalty_init(d::NormalExpert) = [2.0 2.0]
# no_penalty_init(d::NormalExpert) = [1.0 1.0]
# penalize(d::NormalExpert, p) = -0.5 * (p[1] - 1) / (d.σ * d.σ) - (p[2] - 1) * log(d.σ)

## statistics
mean(d::NormalExpert) = mean(Distributions.Normal(d.μ, d.σ))
var(d::NormalExpert) = var(Distributions.Normal(d.μ, d.σ))
quantile(d::NormalExpert, p) = quantile(Distributions.Normal(d.μ, d.σ), p)
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
#     penalty=true, pen_pararms_jk=[1.0 1.0])

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

#     demominator = penalty ? (sum(term_zkz)[1] + (pen_pararms_jk[2] - 1)) : sum(term_zkz)[1]
#     numerator = if penalty
#         (
#             sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
#             (μ_new)^2 * sum(term_zkz)[1] + (pen_pararms_jk[1] - 1)
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
function EM_M_expert_exact(d::NormalExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 1.0])

    # Further E-Step
    Y_e_obs = ye
    Y_sq_e_obs = ye .^ 2

    # Update parameters
    # pos_idx = (ye .!= 0.0)
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    term_zkz = collect(skipmissing(z_e_obs))
    term_zkz_Y = collect(skipmissing(z_e_obs .* Y_e_obs))
    term_zkz_Y_sq = collect(skipmissing(z_e_obs .* Y_sq_e_obs))    

    μ_new = sum(term_zkz_Y)[1] / sum(term_zkz)[1]

    demominator = penalty ? (sum(term_zkz)[1] + (pen_pararms_jk[2] - 1)) : sum(term_zkz)[1]
    numerator = if penalty
        (
            sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
            (μ_new)^2 * sum(term_zkz)[1] + (pen_pararms_jk[1] - 1)
        )
    else
        (
            sum(term_zkz_Y_sq)[1] - 2.0 * μ_new * sum(term_zkz_Y)[1] +
                (μ_new)^2 * sum(term_zkz)[1]
        )
    end
    tmp = numerator / demominator
    σ_new = sqrt(maximum([0.0, tmp]))

    return NormalExpert(μ_new, σ_new)
end