"""
    VonMisesExpert(μ, κ)

PDF:

```math
    f(x; \\mu, \\kappa) = \frac{1}{2 \\pi I_{0}(\\kappa)}
\\exp \\left( \\kappa - cos(x - \\mu)\\right), \\kappa > 0
```

modified Bessels function and their ratio A(k)
[A new type of sharp bounds for ratios of modified Bessel functions](https://www.sciencedirect.com/science/article/pii/S0022247X16302402)
require: 
import Pkg; Pkg.add("Bessels")
using Bessels

See also: [von Mises Distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution)

"""

struct VonMisesExpert{T<:Real} <: RealContinuousExpert
    μ::T
    κ::T
    VonMisesExpert{T}(µ::T, κ::T) where {T<:Real} = new{T}(µ, κ)
end

function VonMisesExpert(μ::T, κ::T; check_args=true) where {T<:Real}
    check_args && @check_args(VonMisesExpert, κ >= zero(κ)) # scale parameter κ > 0 
    return VonMisesExpert{T}(μ, κ)
end

## Outer constructors
VonMisesExpert(μ::Real, κ::Real) = VonMisesExpert(promote(μ, κ)...)
VonMisesExpert(μ::Integer, κ::Integer) = VonMisesExpert(float(μ), float(κ))
VonMisesExpert() = VonMisesExpert(0.0, 1.0) # uniform distribution

## Conversion
function convert(::Type{VonMisesExpert{T}}, μ::S, κ::S) where {T<:Real, S<:Real}
    return VonMisesExpert(T(μ), T(κ))
end
function convert(::Type{VonMisesExpert{T}}, d::VonMisesExpert{S}) where {T<:Real, S<:Real}
    return VonMisesExpert(T(d.μ), T(d.κ); check_args=false)
end
copy(d::VonMisesExpert) = VonMisesExpert(d.μ, d.κ; check_args=false)


function _map_vonmises_domain(x, μ)
    if x > μ + π
        diff = x - (μ + π)
        period = diff ÷ 2π
        return x - (period+1) * 2π
    elseif x < μ - π
        diff = (μ - π) - x
        period = diff ÷ 2π
        return x + (period+1) * 2π
    else
        return x
    end
end

function map_vonmises_domain(x, μ)
    return _map_vonmises_domain.(x, Ref(μ))
end


## Loglikelihood of Expert
function logpdf(d::VonMisesExpert, x...)
    return Distributions.logpdf.(Distributions.VonMises(d.μ, d.κ), map_vonmises_domain(x, d.μ)...)
end
pdf(d::VonMisesExpert, x...) = Distributions.pdf.(Distributions.VonMises(d.μ, d.κ), map_vonmises_domain(x, d.μ)...)
function logcdf(d::VonMisesExpert, x...)
    return Distributions.logcdf.(Distributions.VonMises(d.μ, d.κ), map_vonmises_domain(x, d.μ)...)
end
cdf(d::VonMisesExpert, x...) = Distributions.cdf.(Distributions.VonMises(d.μ, d.κ), map_vonmises_domain(x, d.μ)...)

## expert_ll, etc
expert_ll_exact(d::VonMisesExpert, x::Real) = HMMToolkit.logpdf(d, x)
function expert_ll(d::VonMisesExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end

exposurize_expert(d::VonMisesExpert; exposure=1) = d

# function A(x)
function A(x::Real)
    @check_args(A, x>=0)
    if x <= 713.9
        y = Bessels.besseli1(x) / Bessels.besseli0(x)
    else
        y = (x/(0.5 + sqrt(x^2 + 0.25)) + x/(0.5 + sqrt(x^2 + 2.25)))/2 # estimate A(x) by (upper bound + lower bound)/2
    end
    return y
end

# construct a index invA_table
# if κ -> 1 VonMises distribution corresponds to uniform distribution so κ value below 1 is dropped
# maximum(diff(invA_table.y)) < 0.005, 
# 0.0049999375 < invA_table.y < 0.999898
x = vcat(collect(0.0 : 0.01 : 5),
         collect(5.1 : 0.1 : 20),
         collect(21.0 : 1.0 : 50),
         collect(55.0 : 5.0 : 200),
         collect(210.0 : 100.0 : 5000))
invA_table = DataFrames.DataFrame([x, map(A, x)], [:x, :y])

# function invA(y) use the table to estimate κ
function invA(y)
    y = abs.(y) # A(x) = -A(x)
    # @check_args(invA, maximum(y) < maximum(invA_table.y) && minimum(y) > 0)
            if size(y, 1) == 1
                x = mean(filter(row -> isapprox(row.y, y, atol = 0.005) == true, invA_table).x) # linear interpolation
            else
                x = mean(invA_table.x[abs.(invA_table.y - y) .<= 0.005])
            end
    return(round(x, digits=6))
end

## Parameters
params(d::VonMisesExpert) = (d.μ, d.κ)
function params_init(y, d::VonMisesExpert)
    y = collect(skipmissing(y))
    # pos_idx = (y .> 0.0)  # VonMises distribution takes negative values as well
    μ_init, κ_init = mean(y), invA(1 - var(y))
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    κ_init = isnan(κ_init) || (κ_init == 0) ? 1.0 : κ_init
    return VonMisesExpert(μ_init, κ_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::VonMisesExpert)
    # p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    # return max(
    #     abs(p_zero - 0.0),
    #     (1 - 0.0) *
    #     HypothesisTests.ksstats(y[y .> 0.0], Distributions.Normal(d.μ, d.σ))[2],
    # )
    return HypothesisTests.ksstats(y, Distributions.VonMises(d.μ, d.κ))[2]
end

## Simulation
sim_expert(d::VonMisesExpert) = Distributions.rand(Distributions.VonMises(d.μ, d.κ), 1)[1]

## penalty
penalty_init(d::VonMisesExpert) = [2.0 2.0]
no_penalty_init(d::VonMisesExpert) = [1.0 1.0]
penalize(d::VonMisesExpert, p) = (p[1] - 1) / log(d.κ) - (p[2] - 1) * d.κ

## statistics
mean(d::VonMisesExpert) = mean(Distributions.VonMises(d.μ, d.κ))
var(d::VonMisesExpert) = var(Distributions.VonMises(d.μ, d.κ))
quantile(d::VonMisesExpert, p) = quantile(Distributions.VonMises(d.μ, d.κ), p)

## EM: M-Step, exact observations
function EM_M_expert_exact(d::VonMisesExpert,
    ye, # exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[2.0 2.0])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    # Further E-Step
    Y_e_obs = ye

    # Update parameters
    term_zkz = z_e_obs
    term_zkz_sin_Y = (z_e_obs .* sin.(Y_e_obs))
    term_zkz_cos_Y = (z_e_obs .* cos.(Y_e_obs))

    μ_new = atan(sum(term_zkz_sin_Y)[1] / sum(term_zkz_cos_Y)[1])
    term_zkz_cos_Y_minus_μ = (z_e_obs .* cos.(Y_e_obs .- μ_new))

    denominator = sum(term_zkz)[1]
    numerator = if penalty
        (
            ((pen_params_jk[1] - 1)./ invA_table.x) .- (pen_params_jk[2] - 1) .+ sum(term_zkz_cos_Y_minus_μ)[1]
        )
    else
        (
            sum(term_zkz_cos_Y_minus_μ)[1]
        )
    end
    tmp = HMMToolkit.invA(numerator ./ denominator)
    κ_new = maximum([0.0, tmp])
    # shifting μ does not affect the estimation of κ in the current iteration
    
    μ_tmp = μ_new
    current_ll = sum(HMMToolkit.logpdf.(HMMToolkit.VonMisesExpert(μ_new, κ_new), ye) .* z_e_obs)

    if (-π <= (μ_new - π) <= π) && (sum(HMMToolkit.logpdf.(HMMToolkit.VonMisesExpert(μ_new - π, κ_new), ye) .* z_e_obs) >= current_ll)
        μ_tmp = μ_new - π
    elseif (-π <= (μ_new + π) <= π) && (sum(HMMToolkit.logpdf.(HMMToolkit.VonMisesExpert(μ_new + π, κ_new), ye) .* z_e_obs) >= current_ll)
        μ_tmp = μ_new + π
    end
    μ_new = μ_tmp

    println("μ $(μ_new), tmp $(tmp), κ $(κ_new)")

    return VonMisesExpert(μ_new, κ_new)
end