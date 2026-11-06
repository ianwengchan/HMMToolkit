"""
    HMMIntervalDesign(n; segments=[1:n], time_intervals=nothing, keep_last=true)

Monitoring design for one trip with `n >= 2` **observations**. Ordered, disjoint
`segments` give usable contiguous observation ranges; candidates and lag histories
cannot cross their boundaries. Filtering still uses the complete trip, including
observations outside these ranges. The first observation has no forecast residual.
Set `keep_last=false` only when the last forecast is also omitted, matching the
existing pseudo-residual interface. For a CTHMM, supply all `n-1` positive elapsed
times in `time_intervals`; they are preserved in simulation and filtering.
Segment boundaries are explicit: no sampling rate or dataset is assumed.
"""
struct HMMIntervalDesign
    n::Int
    segments::Vector{UnitRange{Int}}
    time_intervals::Union{Nothing,Vector{Float64}}
    keep_last::Bool
    function HMMIntervalDesign(n::Integer; segments=[1:n], time_intervals=nothing, keep_last::Bool=true)
        n >= 2 || throw(ArgumentError("a trip requires at least two observations"))
        segs = _HMM_interval_segments(segments, n)
        times = isnothing(time_intervals) ? nothing : Float64.(time_intervals)
        if !isnothing(times)
            length(times) == n - 1 || throw(DimensionMismatch("need n-1 elapsed times"))
            all(t -> isfinite(t) && t > 0, times) || throw(ArgumentError("elapsed times must be finite and positive"))
        end
        new(Int(n), segs, times, keep_last)
    end
end

function _HMM_residual_segments(design::HMMIntervalDesign)
    upper = design.n - !design.keep_last
    result = UnitRange{Int}[]
    for seg in design.segments
        a, b = max(2, first(seg)), min(upper, last(seg))
        a <= b && push!(result, (a - 1):(b - 1))
    end
    result
end

function _HMM_interval_model_check(design, response_list, transition, initial, states, continuous_time)
    k = length(initial)
    k > 0 && size(transition) == (k, k) || throw(DimensionMismatch("transition and initial probabilities"))
    size(states) == (length(response_list), k) || throw(DimensionMismatch("emissions and response list"))
    !isempty(response_list) && length(unique(response_list)) == length(response_list) || throw(ArgumentError("responses must be nonempty and distinct"))
    any(x -> Symbol(x) in (:ID, :time_interval), response_list) && throw(ArgumentError("ID and time_interval are reserved design columns"))
    all(isfinite, transition) && all(isfinite, initial) || throw(ArgumentError("non-finite HMM parameters"))
    all(initial .>= 0) && isapprox(sum(initial), 1; atol=1e-10, rtol=1e-10) || throw(ArgumentError("invalid initial probabilities"))
    if continuous_time
        !isnothing(design.time_intervals) || throw(ArgumentError("CTHMM design requires time_intervals"))
        all(i -> transition[i, i] <= 0 && all(j -> j == i || transition[i, j] >= 0, 1:k), 1:k) &&
            all(isapprox.(vec(sum(transition; dims=2)), 0; atol=1e-10, rtol=1e-10)) || throw(ArgumentError("invalid transition rate matrix"))
    else
        all(transition .>= 0) && all(isapprox.(vec(sum(transition; dims=2)), 1; atol=1e-10, rtol=1e-10)) || throw(ArgumentError("invalid transition probability matrix"))
    end
    nothing
end

function _HMM_interval_frame(design, ID=1)
    times = isnothing(design.time_intervals) ? ones(design.n - 1) : design.time_intervals
    DataFrame(ID=fill(ID, design.n), time_interval=vcat(missing, times))
end

# Fixed-design extension of the package simulators. Emission draws use the
# existing expert simulation methods; there is no detector-specific emission or
# HMM filtering implementation. Latent states are not returned.
function _HMM_simulate_interval_trip(ID, transition, initial, states, design, response_list, continuous_time)
    _HMM_interval_model_check(design, response_list, transition, initial, states, continuous_time)
    df = _HMM_interval_frame(design, ID)
    matrices = continuous_time ? Dict(dt => exp(transition * dt) for dt in unique(design.time_intervals)) : nothing
    observations = Matrix{Float64}(undef, design.n, length(response_list))
    state = rand(Categorical(initial))
    for t in 1:design.n
        for d in eachindex(response_list)
            observations[t, d] = sim_expert(states[d, state])
        end
        if t < design.n
            probabilities = continuous_time ? matrices[design.time_intervals[t]] : transition
            state = rand(Categorical(vec(probabilities[state, :])))
        end
    end
    for (d, response) in enumerate(response_list)
        df[!, response] = observations[:, d]
    end
    df
end

"""
    DTHMM_sim_time_series(ID, P_mat, pi, state_list, design::HMMIntervalDesign;
                         response_list=["response1", ...])

Simulate observations at exactly the supplied monitoring design using existing
expert samplers. Unlike the four-argument convenience method, the trip length is
fixed. Latent states are discarded. Use `Random.seed!` to reproduce simulations.
"""
function DTHMM_sim_time_series(ID, P_mat, initial, state_list, design::HMMIntervalDesign;
                               response_list=["response$d" for d in axes(state_list, 1)])
    _HMM_simulate_interval_trip(ID, P_mat, initial, state_list, design, response_list, false)
end

"""
    CTHMM_sim_time_series(ID, Q_mat, pi, state_list, design::HMMIntervalDesign;
                         response_list=["response1", ...])

Simulate a fixed design using `exp(Q_mat * dt)` for each supplied elapsed time
and the existing expert samplers. Latent states are discarded. The original
four-argument convenience method is unchanged.
"""
function CTHMM_sim_time_series(ID, Q_mat, initial, state_list, design::HMMIntervalDesign;
                               response_list=["response$d" for d in axes(state_list, 1)])
    _HMM_simulate_interval_trip(ID, Q_mat, initial, state_list, design, response_list, true)
end

"""
    HMM_interval_residuals(df, responses, transition, pi, states, design;
                           continuous_time=false)

Compute normal **forecast** pseudo-residuals with HMMToolkit's existing emission
precomputation and forecast-residual functions. `df` must be one trip in observation
order, with `design.n` rows. For CTHMMs, elapsed times come from `design`.
Return `(residuals, segments, observation_indices)`: row 1 of `residuals` maps
to observation 2, and `segments` are in residual coordinates. The same routine
is used for observed, reference, and calibration trips. Non-finite usable
residuals are rejected by embedding/scoring; excluded positions are retained.
"""
function HMM_interval_residuals(df::AbstractDataFrame, responses, transition, initial, states,
                                design::HMMIntervalDesign; continuous_time::Bool=false)
    nrow(df) == design.n || throw(DimensionMismatch("observations and monitoring design"))
    _HMM_interval_model_check(design, responses, transition, initial, states, continuous_time)
    seq = _HMM_interval_frame(design)
    for response in responses
        seq[!, response] = df[!, response]
    end
    # A single artificial group lets us reuse the batch emission precomputations
    # without requiring any particular ID field from the caller.
    emission = CTHMM_precompute_batch_data_emission_prob(seq, responses, states)[1][1]
    upper = CTHMM_precompute_batch_data_emission_cdf_separate(seq, responses, states)[1]
    lower = CTHMM_precompute_batch_data_emission_cdf_lower_separate(seq, responses, states)[1]
    forecast = continuous_time ? CTHMM_forecast_pseudo_residuals : DTHMM_forecast_pseudo_residuals
    residuals = forecast(0, seq, emission, upper, lower, transition, initial; keep_last=Int(design.keep_last))
    (; residuals, segments=_HMM_residual_segments(design), observation_indices=collect(2:(design.n - !design.keep_last)))
end

function _HMM_simulated_residuals(design, responses, transition, initial, states, continuous_time)
    simulator = continuous_time ? CTHMM_sim_time_series : DTHMM_sim_time_series
    observations = simulator(1, transition, initial, states, design; response_list=responses)
    HMM_interval_residuals(observations, responses, transition, initial, states, design; continuous_time)
end

"""
    HMM_interval_reference(designs, responses, transition, pi, states;
                           q=1, simulations=500, continuous_time=false)

Build one common residual reference for a fitted HMM. Independently sample a
design uniformly from `designs` for each simulation, simulate a complete trip,
discard its latent states, and run the same normal forecast-residual pipeline.
Pool all gap-safe lag histories across the bank (weighting history rows equally)
and estimate their mean and sample covariance (divisor `N-1`). This bank must be
separate from penalty calibration. Return `(reference, sampled_designs,
pooled_rows, simulations)`. `sampled_designs` contains indices into `designs`.
"""
function HMM_interval_reference(designs::AbstractVector{<:HMMIntervalDesign}, responses,
                                 transition, initial, states; q::Integer=1,
                                 simulations::Integer=500, continuous_time::Bool=false)
    !isempty(designs) || throw(ArgumentError("reference design bank is empty"))
    simulations >= 1 || throw(ArgumentError("simulations must be positive"))
    q >= 1 || throw(ArgumentError("q must be positive"))
    for design in designs
        _HMM_interval_model_check(design, responses, transition, initial, states, continuous_time)
    end
    sampled_designs = Int[]
    banks = Matrix{Float64}[]
    for _ in 1:simulations
        index = rand(eachindex(designs))
        push!(sampled_designs, index)
        prepared = _HMM_simulated_residuals(designs[index], responses, transition, initial, states, continuous_time)
        push!(banks, HMM_lag_embed(prepared.residuals, q; segments=prepared.segments).values)
    end
    pooled = reduce(vcat, banks)
    size(pooled, 1) > size(pooled, 2) || throw(ArgumentError("insufficient pooled histories for a positive definite reference covariance"))
    reference = HMMIntervalReference(q, vec(mean(pooled; dims=1)), cov(pooled; dims=1))
    (; reference, sampled_designs, pooled_rows=size(pooled, 1), simulations=Int(simulations))
end

"""
Result of complete-search calibration: the fixed reference and monitoring
design, prespecified lengths, independent simulated scan maxima, `alpha`, and
the resulting `penalty`. Reuse only for the same fitted HMM and search design.
"""
struct HMMIntervalCalibration
    reference::HMMIntervalReference
    design::HMMIntervalDesign
    lengths::Vector{Int}
    maxima::Vector{Float64}
    alpha::Float64
    penalty::Float64
end

"""
    HMM_calibrate_interval_penalty(design, responses, transition, pi, states,
                                   reference; lengths, simulations=1999,
                                   alpha=0.05, continuous_time=false)

Simulate independent complete trips from the fitted HMM, preserving observation
count, usable segments, elapsed times (CTHMM), and first/last forecast handling.
Apply the fixed common reference and the **complete** interval scan to each;
record its largest `G`. Calibrate the penalty with [`HMM_interval_penalty`](@ref).
There is no pilot tail transform, empirical-trip recentering, or penalty multiplier.
An empty candidate search has maxima `-Inf` and penalty `Inf` (no detection).
"""
function HMM_calibrate_interval_penalty(design::HMMIntervalDesign, responses, transition,
                                        initial, states, reference::HMMIntervalReference;
                                        lengths, simulations::Integer=1999,
                                        alpha::Real=0.05, continuous_time::Bool=false)
    simulations >= 1 || throw(ArgumentError("simulations must be positive"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be between zero and one"))
    _HMM_interval_model_check(design, responses, transition, initial, states, continuous_time)
    length(responses) * reference.q == length(reference.mean) || throw(DimensionMismatch("reference and response dimensions"))
    lens = _HMM_interval_lengths(lengths, reference.q)
    maxima = Vector{Float64}(undef, simulations)
    for b in 1:simulations
        prepared = _HMM_simulated_residuals(design, responses, transition, initial, states, continuous_time)
        maxima[b] = _HMM_scan_intervals(prepared.residuals, reference, lens, prepared.segments; materialize=false)
    end
    penalty = HMM_interval_penalty(maxima; alpha)
    HMMIntervalCalibration(reference, design, lens, maxima, Float64(alpha), penalty)
end

"""
    HMM_detect_intervals(residuals, calibration::HMMIntervalCalibration)

Detect with the exact reference, lengths, and usable segments from calibration.
Residuals must cover the full forecast horizon, including any excluded positions.
Selected intervals gain inclusive `observation_start`/`observation_stop` columns
and whole-search Monte Carlo `pvalue` values. These p-values compare each score
with the simulated **scan maxima**, not with a single-interval null distribution.
"""
function HMM_detect_intervals(residuals::AbstractMatrix, calibration::HMMIntervalCalibration)
    design = calibration.design
    size(residuals, 1) == design.n - 1 - !design.keep_last || throw(DimensionMismatch("residual horizon differs from calibrated design"))
    result = HMM_detect_intervals(residuals, calibration.reference; lengths=calibration.lengths,
        penalty=calibration.penalty, segments=_HMM_residual_segments(design))
    result.intervals.observation_start = result.intervals.start .+ 1
    result.intervals.observation_stop = result.intervals.stop .+ 1
    result.intervals.pvalue = [(1 + count(>=(g), calibration.maxima)) / (length(calibration.maxima) + 1)
                              for g in result.intervals.G]
    (; result..., penalty=calibration.penalty)
end
