using Test, Random, Statistics, LinearAlgebra, Distributions, DataFrames

if !isdefined(Main, :HMMToolkit)
    include(joinpath(@__DIR__, "..", "src", "HMMToolkit.jl"))
end
using .HMMToolkit

@testset "Zero-inflated CDFs" begin
    for p in (0.0, 0.3, 1.0)
        for (expert, base) in ((ZIGammaExpert(p, 0.5, 1.0), Gamma(0.5, 1.0)),
                               (ZIGammaExpert(p, 2.0, 1.0), Gamma(2.0, 1.0)),
                               (ZILogNormalExpert(p, 0.0, 1.0), LogNormal(0.0, 1.0)))
            for y in (-Inf, -1.0, -0.01, 0.0, 0.01, 0.5, 2.0, Inf)
                expected = y < 0 ? 0.0 : p + (1 - p) * cdf(base, y)
                @test HMMToolkit.cdf(expert, y) ≈ expected
                @test exp(HMMToolkit.logcdf(expert, y)) ≈ expected atol=1e-15
                @test lower_cdf(expert, y) ≈ (y == 0 ? 0.0 : expected)
            end
            @test HMMToolkit.cdf(expert, 0.0) == p
            @test lower_cdf(expert, 0.0) == 0.0
        end
    end
    @test HMMToolkit.logcdf(ZIGammaExpert(0.3, 0.5, 1.0), 0.0) ≈ log(0.3)
    @test lower_cdf(NormalExpert(0.0, 1.0), 0.3) == HMMToolkit.cdf(NormalExpert(0.0, 1.0), 0.3)
end

# A test-only count expert exercises the discrete lower-CDF interface without
# adding count distributions to the public package.
struct TestCountExpert <: HMMToolkit.NonZIDiscreteExpert end
HMMToolkit.cdf(::TestCountExpert, y) = cdf(Poisson(2.0), y)
HMMToolkit.pdf(::TestCountExpert, y) = pdf(Poisson(2.0), y)

@testset "Discrete lower CDF and randomized PIT" begin
    for y in (0.0, 1.0, 2.0, 2.5, 5.0)
        @test lower_cdf(TestCountExpert(), y) ≈ cdf(Poisson(2.0), ceil(y) - 1) atol=1e-15
    end
    Random.seed!(37)
    upper = fill(0.7, 10_000, 2)
    lower = fill(0.2, 10_000, 2)
    u = HMMToolkit.randomized_pit(upper, lower)
    @test all(0.2 .<= u .<= 0.7)
    @test abs(mean(u) - 0.45) < 0.01
    @test abs(var(u) - 0.5^2 / 12) < 0.001
    @test length(unique(u[:, 1])) > 9_900
    @test HMMToolkit.randomized_pit(upper, upper) == upper
end

# Independent enumeration of all state paths, excluding the full observation
# at time t for ordinary residuals and using observations before t for forecasts.
function enumerated_bounds(df, responses, states, transitions, initial)
    n, dimensions, k = nrow(df), length(responses), size(states, 2)
    ordinary_lower, ordinary_upper = zeros(n, dimensions), zeros(n, dimensions)
    forecast_lower, forecast_upper = zeros(n - 1, dimensions), zeros(n - 1, dimensions)
    likelihood = [prod(exp(HMMToolkit.expert_ll_exact(states[d, s], df[t, responses[d]]))
                       for d in 1:dimensions) for t in 1:n, s in 1:k]
    for t in 1:n
        ordinary_weights, forecast_weights = zeros(k), zeros(k)
        for path in Iterators.product(ntuple(_ -> 1:k, n)...)
            prior = initial[path[1]] * prod((transitions[j][path[j], path[j + 1]] for j in 1:(n - 1)); init=1.0)
            ordinary_weights[path[t]] += prior * prod((likelihood[j, path[j]] for j in 1:n if j != t); init=1.0)
            forecast_weights[path[t]] += prior * prod((likelihood[j, path[j]] for j in 1:(t - 1)); init=1.0)
        end
        ordinary_weights ./= sum(ordinary_weights)
        forecast_weights ./= sum(forecast_weights)
        for d in 1:dimensions
            y = df[t, responses[d]]
            upper = [HMMToolkit.cdf(states[d, s], y) for s in 1:k]
            lower = [y == 0 && states[d, s] isa HMMToolkit.ZIContinuousExpert ? 0.0 : upper[s] for s in 1:k]
            ordinary_lower[t, d] = dot(ordinary_weights, lower)
            ordinary_upper[t, d] = dot(ordinary_weights, upper)
            if t > 1
                forecast_lower[t - 1, d] = dot(forecast_weights, lower)
                forecast_upper[t - 1, d] = dot(forecast_weights, upper)
            end
        end
    end
    return ordinary_lower, ordinary_upper, forecast_lower, forecast_upper
end

@testset "CTHMM and DTHMM randomized residuals" begin
    Random.seed!(20260916)
    df = DataFrame(ID=ones(Int, 4), time_interval=[0.5, 1.25, 0.8, missing],
                   y=[0.0, 0.5, 0.0, 2.0], z=[-0.4, 0.7, 0.2, -0.1])
    responses = [:y, :z]
    states = [ZIGammaExpert(0.3, 0.5, 1.0) ZILogNormalExpert(0.6, 0.0, 1.0);
              NormalExpert(0.0, 1.0) NormalExpert(1.0, 0.8)]
    initial = [0.4, 0.6]
    q = [-0.7 0.7; 0.4 -0.4]
    p = [0.8 0.2; 0.3 0.7]
    emissions = CTHMM_precompute_batch_data_emission_prob(df, responses, states)[1][1]
    upper = CTHMM_precompute_batch_data_emission_cdf_separate(df, responses, states)[1]
    lower = CTHMM_precompute_batch_data_emission_cdf_lower_separate(df, responses, states)[1]
    for (prefix, transition, transitions) in ((:CTHMM, q, [exp(q * dt) for dt in skipmissing(df.time_interval)]),
                                               (:DTHMM, p, fill(p, 3)))
        @testset "$prefix" begin
            ordinary_fn = getfield(HMMToolkit, Symbol(prefix, :_ordinary_pseudo_residuals))
            forecast_fn = getfield(HMMToolkit, Symbol(prefix, :_forecast_pseudo_residuals))
            combined_fn = getfield(HMMToolkit, Symbol(prefix, :_pseudo_residuals))
            batch_fn = getfield(HMMToolkit, Symbol(prefix, :_batch_pseudo_residuals))
            anomaly_fn = getfield(HMMToolkit, Symbol(prefix, :_batch_anomaly_indices))
            ol, ou, fl, fu = enumerated_bounds(df, responses, states, transitions, initial)
            within(values, lo, hi) = all(lo .- 1e-13 .<= values .<= hi .+ 1e-13)
            for mode in (0, 1)
                ordinary, forecast = combined_fn(mode, df, emissions, upper, lower, transition, initial)
                if mode == 0
                    @test all(isfinite, ordinary)
                    @test all(isfinite, forecast)
                    ordinary, forecast = cdf.(Normal(), ordinary), cdf.(Normal(), forecast)
                end
                @test within(ordinary, ol, ou)
                @test within(forecast, fl, fu)
                @test ordinary[:, 2] ≈ ou[:, 2]
                @test forecast[:, 2] ≈ fu[:, 2]
                @test all(ol[[1, 3], 1] .< ordinary[[1, 3], 1] .< ou[[1, 3], 1])
            end
            @test within(ordinary_fn(1, df, emissions, upper, lower, transition, initial), ol, ou)
            @test within(forecast_fn(1, df, emissions, upper, lower, transition, initial), fl, fu)
            ordinary, forecast = combined_fn(1, df, emissions, upper, lower, transition, initial; keep_last=0)
            @test size(ordinary) == (3, 2)
            @test size(forecast) == (2, 2)
            for keywords in ((;), (; ZI_list=nothing), (; ZI_list=[:y]))
                ordinary, forecast = batch_fn(1, df, responses, transition, initial, states; keywords...)
                @test within(ordinary[1], ol, ou)
                @test within(forecast[1], fl, fu)
                @test 0 < ordinary[1][1, 1] < ou[1, 1]
            end
            # Existing low-level calls remain valid, including a ZI_list.
            ordinary, forecast = combined_fn(1, df, responses, emissions, upper, transition, initial; ZI_list=[:y])
            @test within(ordinary, ol, ou)
            @test within(forecast, fl, fu)
            @test within(ordinary_fn(1, df, responses, emissions, upper, transition, initial; ZI_list=[:y]), ol, ou)
            @test within(forecast_fn(1, df, responses, emissions, upper, transition, initial; ZI_list=[:y]), fl, fu)
            # The public notebook's anomaly-index call still works.
            @test size(anomaly_fn(df, responses, transition, initial, states; ZI_list=nothing)) == (1, 2)
            grouped = vcat(select(df, Not(:ID)), select(df, Not(:ID)))
            grouped.session = repeat([1, 2], inner=4)
            ordinary, forecast = batch_fn(1, grouped, responses, transition, initial, states; group_by_col=:session)
            @test length(ordinary) == length(forecast) == 2
            @test all(within(x, ol, ou) for x in ordinary)
            @test all(within(x, fl, fu) for x in forecast)
        end
    end
    # The prior DTHMM batch interface exposed an unused epsilon keyword.
    @test length(DTHMM_batch_pseudo_residuals(1, df, responses, p, initial, states; ϵ=0).ordinary_list) == 1
end
