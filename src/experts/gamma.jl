"""
    GammaExpert(k, θ)

PDF:

```math
f(x; k, \\theta) = \\frac{x^{k-1} e^{-x/\\theta}}{\\Gamma(k) \\theta^k},
\\quad x > 0
```

See also: [Gamma Distribution](https://en.wikipedia.org/wiki/Gamma_distribution) (Wikipedia), shape-scale parameterization 

"""
struct GammaExpert{T<:Real} <: NonZIContinuousExpert
    k::T
    θ::T
    GammaExpert{T}(k::T, θ::T) where {T<:Real} = new{T}(k, θ)
end

function GammaExpert(k::T, θ::T; check_args=true) where {T<:Real}
    check_args && @check_args(GammaExpert, k >= zero(k) && θ > zero(θ))
    return GammaExpert{T}(k, θ)
end

## Outer constructors
GammaExpert(k::Real, θ::Real) = GammaExpert(promote(k, θ)...)
GammaExpert(k::Integer, θ::Integer) = GammaExpert(float(k), float(θ))
GammaExpert() = GammaExpert(2.0, 1.0)

## Conversion
function convert(::Type{GammaExpert{T}}, k::S, θ::S) where {T<:Real,S<:Real}
    return GammaExpert(T(k), T(θ))
end
function convert(::Type{GammaExpert{T}}, d::GammaExpert{S}) where {T<:Real,S<:Real}
    return GammaExpert(T(d.k), T(d.θ); check_args=false)
end
copy(d::GammaExpert) = GammaExpert(d.k, d.θ; check_args=false)

## Loglikelihood of Expert
function logpdf(d::GammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        Distributions.logpdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function pdf(d::GammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        Distributions.pdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function logcdf(d::GammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        Distributions.logcdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function cdf(d::GammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end

## expert_ll, etc
expert_ll_exact(d::GammaExpert, x::Real) = HMMToolkit.logpdf(d, x)
function expert_ll(d::GammaExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end


exposurize_expert(d::GammaExpert; exposure=1) = d

## Parameters
params(d::GammaExpert) = (d.k, d.θ)
function params_init(y, d::GammaExpert)
    y = collect(skipmissing(y))
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    θ_init = σ2 / μ
    k_init = μ / θ_init
    if isnan(θ_init) || isnan(k_init) || isinf(k_init) || isinf(θ_init) || (θ_init == 0)
        return GammaExpert()
    else
        return GammaExpert(k_init, θ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::GammaExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - 0.0),
        (1 - 0.0) * HypothesisTests.ksstats(y[y .> 0.0], Distributions.Gamma(d.k, d.θ))[2],
    )
end

## Simulation
sim_expert(d::GammaExpert) = Distributions.rand(Distributions.Gamma(d.k, d.θ), 1)[1]

## penalty
penalty_init(d::GammaExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::GammaExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::GammaExpert, p)
    return (p[1] - 1) * log(d.k) - d.k / p[2] + (p[3] - 1) * log(d.θ) - d.θ / p[4]
end

## statistics
mean(d::GammaExpert) = mean(Distributions.Gamma(d.k, d.θ))
var(d::GammaExpert) = var(Distributions.Gamma(d.k, d.θ))
quantile(d::GammaExpert, p) = quantile(Distributions.Gamma(d.k, d.θ), p)


function _gamma_k_to_θ(k, sum_term_zkz, sum_term_zkzy;
    penalty=true, pen_params_jk=[1.0 Inf 1.0 Inf])
    if penalty
        hyper1, hyper2 = pen_params_jk[3], pen_params_jk[4]
        if hyper1 == 1.0 && hyper2 == Inf
            # return sum(term_zkzy)[1] / (sum(term_zkz)[1] * k)   
            return sum_term_zkzy / (sum_term_zkz * k)
        end
        quad1, quad2 = (k * sum_term_zkz - (hyper1 - 1))^2, (4 / hyper2) * sum_term_zkzy
        return (hyper2 / 2) * ((hyper1 - 1) - k * sum_term_zkz + sqrt(quad1 + quad2))
    else
        return sum_term_zkzy / (sum_term_zkz * k)
    end
end

function _gamma_optim_k(logk,
    sum_term_zkz, sum_term_zkzy, sum_term_zkzlogy;
    penalty=true, pen_params_jk=[1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    k_tmp = exp(logk)

    θ_tmp = _gamma_k_to_θ(
        k_tmp, sum_term_zkz, sum_term_zkzy; penalty=penalty, pen_params_jk=pen_params_jk
    )

    obj =
        (k_tmp - 1) * sum_term_zkzlogy - (1 / θ_tmp) * sum_term_zkzy -
        (k_tmp * log(θ_tmp) + loggamma(k_tmp)) * sum_term_zkz
    p = if penalty
        (pen_params_jk[1] - 1) * log(k_tmp) - k_tmp / pen_params_jk[2] +
        (pen_params_jk[3] - 1) * log(θ_tmp) - θ_tmp / pen_params_jk[4]
    else
        0.0
    end
    return (obj + p) * (-1.0)
end


## EM: M-Step, exact observations
function EM_M_expert_exact(d::GammaExpert,
    ye, # exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[1.0 Inf 1.0 Inf])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    # Further E-Step
    Y_e_obs = ye
    logY_e_obs = log.(ye)
    # nan2num(logY_e_obs, 0.0) # get rid of NaN

    # Update parameters
    pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs[pos_idx]
    term_zkz_Y = z_e_obs[pos_idx] .* Y_e_obs[pos_idx]
    term_zkz_logY = z_e_obs[pos_idx] .* logY_e_obs[pos_idx]

    logk_new = Optim.minimizer(
        Optim.optimize(
            x -> _gamma_optim_k(x, sum(term_zkz)[1], sum(term_zkz_Y)[1],
                sum(term_zkz_logY)[1];
                penalty=penalty, pen_params_jk=pen_params_jk),
            max(log(d.k) - 2.0, 0.0), log(d.k) + 2.0),
    )

    k_new = exp(logk_new)
    θ_new = _gamma_k_to_θ(
        k_new,
        sum(term_zkz)[1],
        sum(term_zkz_Y)[1];
        penalty = penalty,
        pen_params_jk = pen_params_jk,
    )

    θ_new = isnan(θ_new) ? 0.0 : θ_new
    θ_new = (maximum([1e-5, θ_new]))

    return GammaExpert(k_new, θ_new)
end