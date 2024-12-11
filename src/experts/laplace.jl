"""
    LaplaceExpert(μ, b)

PDF:

```
 f(x;\\mu,θ) = \\frac{1}{2\\θ}exp(-\\frac{|\\x - \\mu|}{\\θ}), \\θ>0  
```

See also: [Laplace Distribution](https://en.wikipedia.org/wiki/Laplace_distribution) (Wikipedia)

"""

struct LaplaceExpert{T<:Real} <: RealContinuousExpert
    μ::T
    θ::T
    LaplaceExpert{T}(µ::T, θ::T) where {T<:Real} = new{T}(µ, θ)
end

function LaplaceExpert(μ::T, θ::T; check_args=true) where {T<:Real}
    check_args && @check_args(LaplaceExpert, θ >= zero(θ))
    return LaplaceExpert{T}(μ, θ)
end

## Outer constructors
LaplaceExpert(μ::Real, θ::Real) = LaplaceExpert(promote(μ, θ)...)
LaplaceExpert(μ::Integer, θ::Integer) = LaplaceExpert(float(μ), float(θ))
LaplaceExpert() = LaplaceExpert(0.0, 1.0)

## Conversion
function convert(::Type{LaplaceExpert{T}}, μ::S, θ::S) where {T<:Real, S<:Real}
    return LaplaceExpert(T(μ), T(θ))
end
function convert(::Type{LaplaceExpert{T}}, d::LaplaceExpert{S}) where {T<:Real, S<:Real}
    return LaplaceExpert(T(d.μ), T(d.θ); check_args=false)
end
copy(d::LaplaceExpert) = LaplaceExpert(d.μ, d.θ; check_args=false)

## Loglikelihood of Expert
function logpdf(d::LaplaceExpert, x...)
    return Distributions.logpdf.(Distributions.Laplace(d.μ, d.θ), x...)
end
pdf(d::LaplaceExpert, x...) = Distributions.pdf.(Distributions.Laplace(d.μ, d.θ), x...)
function logcdf(d::LaplaceExpert, x...)
    return Distributions.logcdf.(Distributions.Laplace(d.μ, d.θ), x...)
end
cdf(d::LaplaceExpert, x...) = Distributions.cdf.(Distributions.Laplace(d.μ, d.θ), x...)

## expert_ll, etc
expert_ll_exact(d::LaplaceExpert, x::Real) = HMMToolkit.logpdf(d, x)
function expert_ll(d::LaplaceExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end
# function expert_tn(d::LaplaceExpert, tl::Real, yl::Real, yu::Real, tu::Real)
#     expert_tn = if (tl == tu)
#         logpdf.(d, tl)
#     else
#         logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
#     end
#     expert_tn = (tu == 0.0) ? -Inf : expert_tn
#     return expert_tn
# end
# function expert_tn_bar(d::LaplaceExpert, tl::Real, yl::Real, yu::Real, tu::Real)
#     expert_tn_bar = if (tl == tu)
#         0.0
#     else
#         log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))
#     end
#     return expert_tn_bar
# end


exposurize_expert(d::LaplaceExpert; exposure=1) = d

## Parameters
params(d::LaplaceExpert) = (d.μ, d.θ)
function params_init(y, d::LaplaceExpert)
    y = collect(skipmissing(y))
    # pos_idx = (y .> 0.0)  # Laplace distribution takes negative values as well
    μ_init, θ_init = mean(y), sqrt(var(y)/2)
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    θ_init = isnan(θ_init) ? 1.0 : θ_init
    return LaplaceExpert(μ_init, θ_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::LaplaceExpert)
    # p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    # return max(
    #     abs(p_zero - 0.0),
    #     (1 - 0.0) *
    #     HypothesisTests.ksstats(y[y .> 0.0], Distributions.Laplace(d.μ, d.σ))[2],
    # )
    return HypothesisTests.ksstats(y, Distributions.Laplace(d.μ, d.θ))[2]
end

## Simulation
sim_expert(d::LaplaceExpert) = Distributions.rand(Distributions.Laplace(d.μ, d.θ), 1)[1]

## penalty
penalty_init(d::LaplaceExpert) = [2.0 2.0]
no_penalty_init(d::LaplaceExpert) = [1.0 1.0]
penalize(d::LaplaceExpert, p) = - (p[1] - 1) / d.θ - (p[2] - 1) * log(d.θ)

## statistics
mean(d::LaplaceExpert) = mean(Distributions.Laplace(d.μ, d.θ))
var(d::LaplaceExpert) = var(Distributions.Laplace(d.μ, d.θ))
quantile(d::LaplaceExpert, p) = quantile(Distributions.Laplace(d.μ, d.θ), p)

## EM: M-Step, exact observations
function EM_M_expert_exact(d::LaplaceExpert,
    ye, #exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[1.0 1.0])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    # Further E-Step
    Y_e_obs = ye

    # Update parameters
    # pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs

    μ_new = HMMToolkit.weighted_median(ye, z_e_obs)
    term_zkz_Y_minus_μ_abs = abs.(Y_e_obs .- μ_new) .* z_e_obs 

    denominator = penalty ? (sum(term_zkz)[1] + (pen_params_jk[2] - 1)) : sum(term_zkz)[1]
    numerator = penalty ? (sum(term_zkz_Y_minus_μ_abs)[1] + (pen_params_jk[1] - 1)) : sum(term_zkz_Y_minus_μ_abs)[1]
    tmp = numerator / denominator
    θ_new = maximum([0.0, tmp])

    return LaplaceExpert(μ_new, θ_new)
end