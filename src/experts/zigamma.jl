struct ZIGammaExpert{T<:Real} <: ZIContinuousExpert
    p::T
    k::T
    θ::T
    ZIGammaExpert{T}(p::T, k::T, θ::T) where {T<:Real} = new{T}(p, k, θ)
end

function ZIGammaExpert(p::T, k::T, θ::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZIGammaExpert, 0 <= p <= 1 && k >= zero(k) && θ > zero(θ))
    return ZIGammaExpert{T}(p, k, θ)
end

#### Outer constructors
ZIGammaExpert(p::Real, k::Real, θ::Real) = ZIGammaExpert(promote(p, k, θ)...)
function ZIGammaExpert(p::Integer, k::Integer, θ::Integer)
    return ZIGammaExpert(float(p), float(k), float(θ))
end
ZIGammaExpert() = ZIGammaExpert(0.50, 1.0, 1.0)

## Conversion
function convert(::Type{ZIGammaExpert{T}}, p::S, k::S, θ::S) where {T<:Real,S<:Real}
    return ZIGammaExpert(T(p), T(k), T(θ))
end
function convert(::Type{ZIGammaExpert{T}}, d::ZIGammaExpert{S}) where {T<:Real,S<:Real}
    return ZIGammaExpert(T(d.p), T(d.k), T(d.θ); check_args=false)
end
copy(d::ZIGammaExpert) = ZIGammaExpert(d.p, d.k, d.θ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        Distributions.logpdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function pdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        Distributions.pdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function logcdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        # Distributions.logcdf.(Distributions.Gamma(d.k, d.θ), x...)
        log(d.p + (1 - d.p) * Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...))
    end
end
function cdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        # Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...)
        d.p + (1 - d.p) * Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end

## expert_ll, etc
function expert_ll_exact(d::ZIGammaExpert, x::Real)
    return (x == 0.0) ? log(p_zero(d)) : log(1 - p_zero(d)) + HMMToolkit.logpdf(d, x)
end
function expert_ll(d::ZIGammaExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = HMMToolkit.expert_ll(HMMToolkit.GammaExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = if (yl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_ll_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_ll_pos))
    end
    expert_ll = (tu == 0.0) ? log.(p0) : expert_ll
    return expert_ll
end


exposurize_expert(d::ZIGammaExpert; exposure=1) = d

## Parameters
params(d::ZIGammaExpert) = (d.p, d.k, d.θ)
p_zero(d::ZIGammaExpert) = d.p
function params_init(y, d::ZIGammaExpert)
    y = collect(skipmissing(y))
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    θ_init = σ2 / μ
    k_init = μ / θ_init
    if isnan(θ_init) || isnan(k_init) || isinf(k_init) || isinf(θ_init) || (θ_init == 0)
        return ZIGammaExpert()
    else
        return ZIGammaExpert(p_init, k_init, θ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::ZIGammaExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - d.p),
        (1 - d.p) * HypothesisTests.ksstats(y[y .> 0.0], Distributions.Gamma(d.k, d.θ))[2],
    )
end

## Simulation
function sim_expert(d::ZIGammaExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(Distributions.Gamma(d.k, d.θ), 1)[1]
end

## penalty
penalty_init(d::ZIGammaExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIGammaExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::ZIGammaExpert, p)
    return (p[1] - 1) * log(d.k) - d.k / p[2] + (p[3] - 1) * log(d.θ) - d.θ / p[4]
end

## statistics
mean(d::ZIGammaExpert) = (1 - d.p) * mean(Distributions.Gamma(d.k, d.θ))
function var(d::ZIGammaExpert)
    return (1 - d.p) * var(Distributions.Gamma(d.k, d.θ)) +
           d.p * (1 - d.p) * (mean(Distributions.Gamma(d.k, d.θ)))^2
end
function quantile(d::ZIGammaExpert, p)
    return p <= d.p ? 0.0 : quantile(Distributions.Gamma(d.k, d.θ), p - d.p)
end


## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIGammaExpert,
    ye, # exposure,
    z_e_obs;
    penalty=true, pen_params_jk=[1.0 Inf 1.0 Inf])

    # Remove missing values first
    ## turn z_e_obs of missing data to missing as well, to apply skipmissing below
    z_e_obs = [ismissing(x) ? missing : y for (x, y) in zip(ye, z_e_obs)]
    ye = collect(skipmissing(ye))
    z_e_obs = collect(skipmissing(z_e_obs))

    # Old parameters
    k_old = d.k
    θ_old = d.θ
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    if p_old < 1e-6
        p_old = 1e-6
    end

    # Update zero probability
    expert_ll_pos = HMMToolkit.expert_ll_exact.(HMMToolkit.GammaExpert(d.k, d.θ), ye)

    z_zero_e_obs = z_e_obs .* HMMToolkit.EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = HMMToolkit.EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    p_new = max(0.0, min(1 - 1e-08, p_new))

    # Update parameters: call its positive part
    tmp_exp = GammaExpert(k_old, θ_old)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye, # exposure,
        z_pos_e_obs;
        penalty=penalty, pen_params_jk=pen_params_jk)

    return ZIGammaExpert(p_new, tmp_update.k, tmp_update.θ)
end