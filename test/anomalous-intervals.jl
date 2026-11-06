using Test, Random, Statistics, LinearAlgebra, Distributions, DataFrames

@testset "Lag embeddings and interval moments" begin
    z = [1.0 10; 2 20; 3 30; 4 40; 5 50; 6 60]
    embedded = HMM_lag_embed(z, 2; segments=[1:3, 5:6])
    @test embedded.positions == [2, 3, 6]
    @test embedded.values == [1 10 2 20; 2 20 3 30; 5 50 6 60]
    @test HMM_lag_embed(z, 1).values == z
    @test isempty(HMM_lag_embed(z, 4; segments=[1:3, 5:6]).positions)
    @test_throws ArgumentError HMM_lag_embed(z, 0)
    @test_throws ArgumentError HMM_lag_embed(z, 1; segments=[1:3, 3:6])
    @test_throws ArgumentError HMM_lag_embed(z, 1; segments=[2:7])
    @test_throws ArgumentError HMM_lag_embed(fill(Inf, 2, 1), 1)
    @test_throws ArgumentError HMMIntervalReference(2, zeros(3), Matrix{Float64}(I, 3, 3))
    @test_throws PosDefException HMMIntervalReference(1, zeros(2), zeros(2, 2))
    values = [1.0 2; 3 4; 5 3]
    score = HMM_interval_score(values)
    mu = vec(mean(values; dims=1))
    sigma = cov(values; dims=1, corrected=false)
    v = (1 - 2/5) * sigma + (2/5) * I
    @test score.mean ≈ mu
    @test score.covariance ≈ sigma
    @test score.regularized_covariance ≈ v
    @test score.shrinkage == 2/5
    @test score.G ≈ 3 * (sum(abs2, mu) + tr(v) - 2 - logdet(v))
    @test score.G ≈ score.G_mean + score.G_cov
    @test HMM_interval_score(zeros(1, 4)).regularized_covariance ≈ 0.8I
    @test_throws ArgumentError HMM_interval_score(zeros(0, 2))
end

@testset "Complete scans agree with direct KL calculations" begin
    Random.seed!(310)
    for p in 1:3, q in 1:3
        z = randn(18, p)
        z[9:10, :] .= NaN
        d = p * q
        a = randn(d, d)
        sigma = a * a' + I
        mu = randn(d)
        reference = HMMIntervalReference(q, mu, sigma)
        segments, lengths = [1:8, 11:18], [q, q+2, q+5]
        scan = HMM_scan_intervals(z, reference; lengths, segments)
        @test nrow(scan) == sum(max(0, length(seg)-len+1) for seg in segments for len in lengths)
        for row in eachrow(scan)
            # Independently stack only histories inside this candidate and use
            # a direct two-pass covariance, rather than the scan's prefixes.
            histories = hcat([z[(row.start+lag):(row.stop-q+1+lag), channel]
                              for lag in 0:q-1 for channel in 1:p]...)
            x = (histories .- mu') / cholesky(Symmetric(sigma)).U
            center = vec(mean(x; dims=1))
            s = (x .- center')' * (x .- center') / size(x, 1)
            gamma = d/(size(x, 1)+d)
            v = (1-gamma)*s + gamma*I
            oracle = size(x, 1)*(dot(center, center)+tr(v)-d-logdet(v))
            @test row.G ≈ oracle atol=1e-9
            @test row.embedded_rows == row.length-q+1
            @test row.shrinkage == gamma
            @test row.G ≈ row.G_mean + row.G_cov
            @test any(seg -> row.start in seg && row.stop in seg, segments)
        end
        @test nrow(HMM_scan_intervals(z, reference; lengths=[30], segments)) == 0
        @test_throws ArgumentError HMM_scan_intervals(z, reference; lengths=[q, q], segments)
    end
    z = randn(12, 2)
    reference = HMMIntervalReference(3, zeros(6), Matrix{Float64}(I, 6, 6))
    before = HMM_scan_intervals(z, reference; lengths=[6])
    z[1:3, :] .+= 10
    after = HMM_scan_intervals(z, reference; lengths=[6])
    @test before.G[before.start .== 4] ≈ after.G[after.start .== 4] atol=1e-9
end

# Exhaustive subset objective, independent of the DP recursion.
function interval_subset_oracle(candidates, penalty)
    best = 0.0
    for mask in 0:(2^nrow(candidates)-1)
        selected = [i for i in 1:nrow(candidates) if (mask >> (i-1)) & 1 == 1]
        sort!(selected; by=i -> candidates.start[i])
        all(candidates.stop[selected[j]] < candidates.start[selected[j+1]]
            for j in 1:(length(selected)-1)) || continue
        best = max(best, sum((candidates.G[i]-penalty for i in selected); init=0.0))
    end
    best
end

@testset "Optimal selection, adjacency, and deterministic ties" begin
    # Greedy would take [1,6] with utility 10, losing the optimal 6+6.
    candidates = DataFrame(start=[1, 1, 4], stop=[6, 3, 6], G=[12., 8, 8])
    result = HMM_select_intervals(candidates, 2.)
    @test result.selected_rows == [2, 3]
    @test result.objective == 12
    @test HMM_select_intervals(DataFrame(start=[1], stop=[3], G=[2.]), 2.).objective == 0
    @test isempty(HMM_select_intervals(candidates, Inf).selected_rows)
    @test HMM_select_intervals(DataFrame(start=[1, 1, 4], stop=[6, 3, 6], G=[12., 7, 7]), 2.).selected_rows == [1]
    @test HMM_select_intervals(DataFrame(start=[2, 1], stop=[5, 4], G=[7., 7]), 2.).selected_rows == [2]
    @test_throws ArgumentError HMM_select_intervals(candidates, -1)
    Random.seed!(518)
    for _ in 1:60
        starts = rand(1:12, 10)
        input = DataFrame(start=starts, stop=starts .+ rand(0:5, 10), G=Float64.(rand(0:12, 10)))
        penalty = rand(0:5)
        answer = HMM_select_intervals(input, penalty)
        @test answer.objective ≈ interval_subset_oracle(input, penalty)
        @test all(answer.intervals.stop[j] < answer.intervals.start[j+1] for j in 1:nrow(answer.intervals)-1)
    end
end

@testset "Finite Monte Carlo penalty" begin
    @test HMM_interval_penalty(collect(1.:1999.)) == 1900
    @test HMM_interval_penalty(collect(1.:18.)) == Inf
    @test HMM_interval_penalty(ones(1999)) == 1
    @test HMM_interval_penalty(fill(-Inf, 19)) == Inf
    for maxima in (collect(1.:19.), repeat([1., 2., 3.], 13))
        cutoff = HMM_interval_penalty(maxima; alpha=.2)
        for g in unique(vcat(maxima, cutoff+.01))
            pvalue = (1 + count(>=(g), maxima))/(length(maxima)+1)
            @test (g > cutoff) == (pvalue <= .2)
        end
    end
    @test_throws ArgumentError HMM_interval_penalty(Float64[])
    @test_throws ArgumentError HMM_interval_penalty([NaN])
    @test_throws ArgumentError HMM_interval_penalty([1.]; alpha=0)
end

function enumerated_forecasts(observations, responses, transitions, initial, states)
    n, k, p = nrow(observations), length(initial), length(responses)
    expected = zeros(n-1, p)
    for t in 2:n
        weights = zeros(k)
        for path in Iterators.product(ntuple(_ -> 1:k, t)...)
            probability = initial[path[1]]
            for u in 1:t-1
                probability *= transitions[u][path[u], path[u+1]]
                for d in 1:p
                    expert = states[d, path[u]]
                    probability *= pdf(Normal(expert.μ, expert.σ), observations[u, responses[d]])
                end
            end
            weights[path[t]] += probability
        end
        weights ./= sum(weights)
        for d in 1:p
            uniform = sum(weights[s]*cdf(Normal(states[d,s].μ, states[d,s].σ), observations[t,responses[d]]) for s in 1:k)
            expected[t-1,d] = quantile(Normal(), uniform)
        end
    end
    expected
end

@testset "HMM residual reuse and fixed monitoring designs" begin
    p, rate, initial = [.85 .15; .25 .75], [-.4 .4; .7 -.7], [.4, .6]
    states = [NormalExpert(-.5,.8) NormalExpert(.7,1.1);
              NormalExpert(.3,1.2) NormalExpert(-.8,.7)]
    responses = [:lateral, :longitudinal]
    observations = DataFrame(lateral=[.2,-.3,.1,.7,.4], longitudinal=[-.1,.5,.2,-.6,.1])
    times = [.2, .7, 1.1, .3]
    for ct in (false, true)
        transition = ct ? rate : p
        design = HMMIntervalDesign(5; segments=[1:3, 4:5], time_intervals=ct ? times : nothing)
        prepared = HMM_interval_residuals(observations, responses, transition, initial, states, design; continuous_time=ct)
        matrices = ct ? [exp(rate*dt) for dt in times] : fill(p, 4)
        @test prepared.residuals ≈ enumerated_forecasts(observations, responses, matrices, initial, states) atol=1e-12
        @test prepared.observation_indices == [2,3,4,5]
        @test prepared.segments == [1:2,3:4]
        shortened = HMMIntervalDesign(5; segments=[1:3,4:5], time_intervals=ct ? times : nothing, keep_last=false)
        trimmed = HMM_interval_residuals(observations, responses, transition, initial, states, shortened; continuous_time=ct)
        @test trimmed.residuals ≈ prepared.residuals[1:end-1,:]
        @test trimmed.segments == [1:2,3:3]
        simulator = ct ? CTHMM_sim_time_series : DTHMM_sim_time_series
        simulated = simulator(7, transition, initial, states, design; response_list=responses)
        @test nrow(simulated) == 5
        @test all(simulated.ID .== 7)
        @test !(:true_state in propertynames(simulated))
        @test collect(skipmissing(simulated.time_interval)) == (ct ? times : ones(4))
    end
    @test_throws ArgumentError HMMIntervalDesign(5; time_intervals=[0.,1,1,1])
    @test_throws ArgumentError HMM_interval_residuals(observations, responses, rate, initial, states, HMMIntervalDesign(5); continuous_time=true)
    # Original convenience simulators remain callable.
    @test 30 <= nrow(DTHMM_sim_time_series(1, p, initial, states)) <= 150
    pts = HMMToolkit.CTHMM_precompute_distinct_time_Pt_list(collect(1:30), rate)
    @test 30 <= nrow(CTHMM_sim_time_series(1, pts, initial, states)) <= 150
end

@testset "Independent reference bank and complete-scan calibration" begin
    p, initial = [.9 .1; .2 .8], [.5, .5]
    states = [NormalExpert(-.4,.9) NormalExpert(.8,1.1);
              NormalExpert(.2,1.2) NormalExpert(-.5,.8)]
    responses = [:a,:b]
    designs = [HMMIntervalDesign(22; segments=[1:10,13:22]), HMMIntervalDesign(27; keep_last=false)]
    Random.seed!(811)
    bank = HMM_interval_reference(designs, responses, p, initial, states; q=2, simulations=8)
    # Reconstruct the pooled bank independently using only public API calls.
    Random.seed!(811)
    pooled, chosen = Matrix{Float64}[], Int[]
    for _ in 1:8
        i = rand(eachindex(designs)); push!(chosen, i)
        df = DTHMM_sim_time_series(1,p,initial,states,designs[i]; response_list=responses)
        prepared = HMM_interval_residuals(df,responses,p,initial,states,designs[i])
        push!(pooled,HMM_lag_embed(prepared.residuals,2;segments=prepared.segments).values)
    end
    values = reduce(vcat,pooled)
    @test bank.sampled_designs == chosen
    @test bank.pooled_rows == size(values,1)
    @test bank.reference.mean ≈ vec(mean(values;dims=1))
    @test bank.reference.covariance ≈ cov(values;dims=1)
    whitened = (values .- bank.reference.mean') / bank.reference.chol'
    @test vec(mean(whitened;dims=1)) ≈ zeros(4) atol=1e-12
    @test cov(whitened;dims=1) ≈ Matrix{Float64}(I,4,4) atol=1e-12
    design = designs[1]
    Random.seed!(912)
    calibration = HMM_calibrate_interval_penalty(design,responses,p,initial,states,bank.reference;
        lengths=[4,6],simulations=19)
    Random.seed!(912)
    maxima = Float64[]
    for _ in 1:19
        df = DTHMM_sim_time_series(1,p,initial,states,design;response_list=responses)
        prepared = HMM_interval_residuals(df,responses,p,initial,states,design)
        scan = HMM_scan_intervals(prepared.residuals,bank.reference;lengths=[4,6],segments=prepared.segments)
        push!(maxima,maximum(scan.G))
    end
    @test calibration.maxima ≈ maxima
    @test calibration.penalty == maximum(maxima)
    result = HMM_detect_intervals(fill(4.,21,2),calibration)
    @test !isempty(result.intervals)
    @test result.intervals.observation_start == result.intervals.start .+ 1
    @test result.intervals.observation_stop == result.intervals.stop .+ 1
    @test all(result.intervals.pvalue .<= .05)
    @test all(result.intervals.G .> calibration.penalty)
    @test all(length(row.mean)==4 && size(row.covariance)==(4,4) for row in eachrow(result.intervals))
    @test_throws DimensionMismatch HMM_detect_intervals(zeros(20,2),calibration)
    # The full CTHMM reference/calibration path preserves irregular elapsed times.
    ct = HMMIntervalDesign(15;time_intervals=collect(range(.1,.8;length=14)),segments=[1:7,9:15])
    rate = [-.4 .4; .7 -.7]
    ctbank = HMM_interval_reference([ct],responses,rate,initial,states;simulations=3,continuous_time=true)
    ctcal = HMM_calibrate_interval_penalty(ct,responses,rate,initial,states,ctbank.reference;
        lengths=[3,5],simulations=19,continuous_time=true)
    @test length(ctcal.maxima)==19 && isfinite(ctcal.penalty)
    emptycal = HMM_calibrate_interval_penalty(design,responses,p,initial,states,bank.reference;lengths=[100],simulations=2)
    @test all(==(-Inf),emptycal.maxima)
    @test emptycal.penalty == Inf
    @test isempty(HMM_detect_intervals(zeros(21,2),emptycal).intervals)
end
