"""
    NormalExpert(μ, σ)

PDF:

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\x - \\mu)^2}{2 \\sigma^2} \\right)
```

See also: [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution) (Wikipedia)

"""
struct NormalExpert{T<:Real} <: RealContinuousExpert
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
expert_ll_exact(d::NormalExpert, x::Real) = HMMToolkit.logpdf(d, x)
function expert_ll(d::NormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end


exposurize_expert(d::NormalExpert; exposure=1) = d

## Parameters
params(d::NormalExpert) = (d.μ, d.σ)
function params_init(y, d::NormalExpert)
    y = collect(skipmissing(y))
    # pos_idx = (y .> 0.0)  # Normal distribution takes negative values as well
    μ_init, σ_init = mean(y), sqrt(var(y))
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    σ_init = isnan(σ_init) || (σ_init == 0) ? 1.0 : σ_init
    return NormalExpert(μ_init, σ_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::NormalExpert)
    # p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    # return max(
    #     abs(p_zero - 0.0),
    #     (1 - 0.0) *
    #     HypothesisTests.ksstats(y[y .> 0.0], Distributions.Normal(d.μ, d.σ))[2],
    # )
    return HypothesisTests.ksstats(y, Distributions.Normal(d.μ, d.σ))[2]
end

## Simulation
sim_expert(d::NormalExpert) = Distributions.rand(Distributions.Normal(d.μ, d.σ), 1)[1]

## penalty
penalty_init(d::NormalExpert) = [2.0 2.0]
no_penalty_init(d::NormalExpert) = [1.0 1.0]
penalize(d::NormalExpert, p) = -0.5 * (p[1] - 1) / (d.σ * d.σ) - (p[2] - 1) * log(d.σ)

## statistics
mean(d::NormalExpert) = mean(Distributions.Normal(d.μ, d.σ))
var(d::NormalExpert) = var(Distributions.Normal(d.μ, d.σ))
quantile(d::NormalExpert, p) = quantile(Distributions.Normal(d.μ, d.σ), p)


## EM: M-Step, exact observations
function EM_M_expert_exact(
    d::NormalExpert,
    ye, # exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[1.0 1.0])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    # Further E-Step
    Y_e_obs = ye
    Y_sq_e_obs = ye .^ 2

    # Update parameters
    # pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs
    term_zkz_Y = (z_e_obs .* Y_e_obs)
    term_zkz_Y_sq = (z_e_obs .* Y_sq_e_obs)

    μ_new = sum(term_zkz_Y)[1] / sum(term_zkz)[1]

    denominator = penalty ? (sum(term_zkz)[1] + (pen_params_jk[2] - 1)) : sum(term_zkz)[1]
    numerator = if penalty
        (
            sum(term_zkz_Y_sq)[1] - 2.0 * μ_new * sum(term_zkz_Y)[1] +
            (μ_new)^2 * sum(term_zkz)[1] + (pen_params_jk[1] - 1)
        )
    else
        (
            sum(term_zkz_Y_sq)[1] - 2.0 * μ_new * sum(term_zkz_Y)[1] +
            (μ_new)^2 * sum(term_zkz)[1]
        )
    end
    tmp = numerator / denominator
    tmp = isnan(tmp) ? 0.0 : tmp

    σ_new = sqrt(maximum([1e-10, tmp]))

    return NormalExpert(μ_new, σ_new)
end