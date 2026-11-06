"""
    HMMIntervalReference(q, mean, covariance)

Common model-generated moments for a `p*q` dimensional, oldest-to-newest lag
embedding of normal forecast pseudo-residuals. `covariance` must be symmetric
positive definite. Estimate these moments from an independent simulation bank
with [`HMM_interval_reference`](@ref), never from the trip being scanned.
"""
struct HMMIntervalReference
    q::Int
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    chol::LowerTriangular{Float64,Matrix{Float64}}
    function HMMIntervalReference(q::Integer, mean::AbstractVector, covariance::AbstractMatrix)
        q >= 1 || throw(ArgumentError("q must be positive"))
        d = length(mean)
        d > 0 && d % q == 0 || throw(ArgumentError("reference dimension must be a positive multiple of q"))
        size(covariance) == (d, d) || throw(DimensionMismatch("reference covariance dimension"))
        all(isfinite, mean) && all(isfinite, covariance) || throw(ArgumentError("non-finite reference moments"))
        issymmetric(covariance) || throw(ArgumentError("reference covariance must be symmetric"))
        sigma = Matrix{Float64}(covariance)
        new(Int(q), Vector{Float64}(mean), sigma, cholesky(Symmetric(sigma)).L)
    end
end

function _HMM_interval_segments(segments, n)
    result = UnitRange{Int}[]
    previous = 0
    for segment in segments
        segment isa AbstractUnitRange{<:Integer} || throw(ArgumentError("segments must be integer unit ranges"))
        isempty(segment) && throw(ArgumentError("segments must be nonempty"))
        1 <= first(segment) <= last(segment) <= n || throw(ArgumentError("segment outside residual horizon"))
        first(segment) > previous || throw(ArgumentError("segments must be ordered and disjoint"))
        push!(result, Int(first(segment)):Int(last(segment)))
        previous = last(segment)
    end
    result
end

"""
    HMM_lag_embed(residuals, q; segments=[1:size(residuals, 1)])

Stack `q` consecutive rows of the `n × p` normal residual matrix, oldest first.
Return `(values, positions)`, where `positions` gives the residual index at the
end of each history. Histories never cross a segment boundary. Segments describe
usable contiguous residual positions; excluded positions may be non-finite.
No interpolation, imputation, or clipping of residuals is performed.
"""
function HMM_lag_embed(residuals::AbstractMatrix, q::Integer; segments=[1:size(residuals, 1)])
    q >= 1 || throw(ArgumentError("q must be positive"))
    n, p = size(residuals)
    p > 0 || throw(ArgumentError("at least one residual channel is required"))
    segs = _HMM_interval_segments(segments, n)
    positions = [t for seg in segs for t in (first(seg) + q - 1):last(seg)]
    values = Matrix{Float64}(undef, length(positions), p * q)
    for (row, t) in enumerate(positions), lag in 0:(q - 1), channel in 1:p
        value = residuals[t - q + 1 + lag, channel]
        value isa Real && isfinite(value) || throw(ArgumentError("non-finite usable residual at position $(t - q + 1 + lag)"))
        values[row, lag * p + channel] = value
    end
    (; values, positions)
end

"""
    HMM_interval_score(values)

Score already whitened lag-history rows. With `n_I = size(values, 1)` and
`d = p*q`, compute the sample mean, ML sample covariance (divisor `n_I`),
`gamma = d/(n_I+d)`, `V = (1-gamma)*S + gamma*I`, and
`G = n_I*(dot(mean,mean) + tr(V) - d - logdet(V))`.
Return these moments, shrinkage, and the separate mean and covariance scores.
"""
function HMM_interval_score(values::AbstractMatrix)
    n, d = size(values)
    n > 0 && d > 0 || throw(ArgumentError("interval must contain history rows and channels"))
    all(isfinite, values) || throw(ArgumentError("non-finite interval values"))
    mu = vec(mean(values; dims=1))
    centered = values .- transpose(mu)
    covariance = transpose(centered) * centered / n
    shrinkage = d / (n + d)
    regularized_covariance = Matrix(Symmetric((1 - shrinkage) * covariance + shrinkage * I))
    G_mean = n * dot(mu, mu)
    G_cov = max(0.0, n * (tr(regularized_covariance) - d - logdet(cholesky(Symmetric(regularized_covariance)))))
    (; G=G_mean + G_cov, G_mean, G_cov, mean=mu, covariance,
       regularized_covariance, shrinkage, embedded_rows=n, dimension=d)
end

function _HMM_interval_lengths(lengths, q)
    values = collect(lengths)
    !isempty(values) && all(x -> x isa Integer && x >= q, values) ||
        throw(ArgumentError("lengths must be nonempty integers at least q"))
    length(unique(values)) == length(values) || throw(ArgumentError("duplicate candidate lengths"))
    sort!(Int.(values))
end

function _HMM_whiten_histories(residuals, reference, segments)
    size(residuals, 2) * reference.q == length(reference.mean) || throw(DimensionMismatch("residual channels and reference moments"))
    embedded = HMM_lag_embed(residuals, reference.q; segments)
    whitened = (embedded.values .- transpose(reference.mean)) / transpose(reference.chol)
    values = zeros(size(residuals, 1), length(reference.mean))
    values[embedded.positions, :] = whitened
    values
end

function _HMM_empty_scan()
    DataFrame(start=Int[], stop=Int[], length=Int[], embedded_rows=Int[],
        dimension=Int[], shrinkage=Float64[], G=Float64[], G_mean=Float64[], G_cov=Float64[])
end

# Prefix moments let the observed and calibration scans share exactly the same
# enumeration and score, without storing every calibration candidate.
function _HMM_scan_intervals(residuals, reference, lengths, segments; materialize=true)
    segs = _HMM_interval_segments(segments, size(residuals, 1))
    lens = _HMM_interval_lengths(lengths, reference.q)
    x = _HMM_whiten_histories(residuals, reference, segs)
    nobs, d = size(x)
    sums = zeros(nobs + 1, d)
    products = zeros(nobs + 1, d, d)
    for t in 1:nobs, i in 1:d
        sums[t + 1, i] = sums[t, i] + x[t, i]
        for j in 1:i
            products[t + 1, i, j] = products[t, i, j] + x[t, i] * x[t, j]
        end
    end
    result = _HMM_empty_scan()
    largest = -Inf
    mu, regularized = zeros(d), zeros(d, d)
    for seg in segs, len in lens
        n = len - reference.q + 1
        shrinkage = d / (n + d)
        for a in first(seg):(last(seg) - len + 1)
            b = a + len - 1
            history_start = a + reference.q - 1
            for i in 1:d
                mu[i] = (sums[b + 1, i] - sums[history_start, i]) / n
            end
            for i in 1:d, j in 1:i
                s = (products[b + 1, i, j] - products[history_start, i, j]) / n - mu[i] * mu[j]
                regularized[i, j] = regularized[j, i] = (1 - shrinkage) * s + (i == j ? shrinkage : 0.0)
            end
            G_mean = n * dot(mu, mu)
            tracev = tr(regularized)
            logdetv = logdet(cholesky!(Symmetric(regularized)))
            G_cov = max(0.0, n * (tracev - d - logdetv))
            G = G_mean + G_cov
            largest = max(largest, G)
            materialize && push!(result, (a, b, len, n, d, shrinkage, G, G_mean, G_cov))
        end
    end
    materialize ? result : largest
end

"""
    HMM_scan_intervals(residuals, reference; lengths, segments=[1:size(residuals, 1)])

Enumerate every start for each prespecified length, wholly within a usable
segment. Boundaries are inclusive **residual positions**. For `[a,b]`, only
histories ending at `a+q-1:b` enter the score: all lags lie inside the interval,
and `n_I = b-a-q+2`. No boundary extension or minimum gap is imposed.
Return a DataFrame with boundaries, lengths, embedded row counts, dimension,
shrinkage, `G`, `G_mean`, and `G_cov`. An empty search returns the same schema.
"""
function HMM_scan_intervals(residuals::AbstractMatrix, reference::HMMIntervalReference;
                            lengths, segments=[1:size(residuals, 1)])
    _HMM_scan_intervals(residuals, reference, lengths, segments)
end

"""
    HMM_interval_penalty(maxima; alpha=0.05)

Conservative empirical upper-tail cutoff for independent complete-scan maxima.
Use the finite Monte Carlo rank rule `(1 + count(>=(G), maxima))/(B+1) <= alpha`
with strict eligibility `G > penalty`. Ties at the cutoff are excluded. If
`alpha < 1/(B+1)`, return `Inf`; the simulation bank cannot resolve that level.
This is the research implementation's empirical quantile convention, without
interpolation between order statistics.
"""
function HMM_interval_penalty(maxima::AbstractVector; alpha::Real=0.05)
    0 < alpha < 1 || throw(ArgumentError("alpha must be between zero and one"))
    !isempty(maxima) && all(x -> isfinite(x) || x == -Inf, maxima) || throw(ArgumentError("invalid or empty scan maxima"))
    all(==(-Inf), maxima) && return Inf
    allowed = floor(Int, alpha * (length(maxima) + 1) - 1 + 1e-12)
    allowed < 0 && return Inf
    sort(Float64.(maxima); rev=true)[allowed + 1]
end

function _HMM_backtrack(take, predecessor, j)
    picked = Int[]
    while j > 0
        if take[j + 1]
            push!(picked, j)
            j = predecessor[j]
        else
            j -= 1
        end
    end
    reverse!(picked)
end

"""
    HMM_select_intervals(candidates, penalty)

Globally maximize `sum(G - penalty)` over non-overlapping inclusive intervals
using weighted interval scheduling and backtracking. The empty solution is
allowed; only strictly positive utilities are eligible. Adjacent intervals may
touch (`previous.stop == next.start-1`). Numerical ties (relative tolerance
`1e-12`, as in the research code) prefer fewer intervals, then larger total `G`,
then lexicographically earlier endpoints. Return `(intervals, selected_rows,
objective)`, with `utility` added to the selected candidate table.
"""
function HMM_select_intervals(candidates::AbstractDataFrame, penalty::Real)
    all(in(propertynames(candidates)), [:start, :stop, :G]) || throw(ArgumentError("candidates require start, stop, and G"))
    !isnan(penalty) && penalty >= 0 || throw(ArgumentError("penalty must be nonnegative"))
    all(i -> candidates.start[i] isa Integer && candidates.stop[i] isa Integer &&
        1 <= candidates.start[i] <= candidates.stop[i] && isfinite(candidates.G[i]) && candidates.G[i] >= 0,
        1:nrow(candidates)) || throw(ArgumentError("invalid interval boundary or score"))
    rows = findall(candidates.G .> penalty)
    sort!(rows; by=i -> (candidates.stop[i], candidates.start[i]))
    ends = candidates.stop[rows]
    n = length(rows)
    predecessor = [searchsortedlast(@view(ends[1:j-1]), candidates.start[rows[j]] - 1) for j in 1:n]
    utility, total = zeros(n + 1), zeros(n + 1)
    counts, take = zeros(Int, n + 1), falses(n + 1)
    for j in 1:n
        row, base = rows[j], predecessor[j] + 1
        added = utility[base] + candidates.G[row] - penalty
        added_count = counts[base] + 1
        added_total = total[base] + candidates.G[row]
        tolerance = 1e-12 * max(1.0, abs(added), abs(utility[j]))
        choose = added > utility[j] + tolerance
        if abs(added - utility[j]) <= tolerance
            if added_count != counts[j]
                choose = added_count < counts[j]
            elseif abs(added_total - total[j]) > 1e-12 * max(1.0, abs(added_total), abs(total[j]))
                choose = added_total > total[j]
            else
                with_j = vcat(_HMM_backtrack(take, predecessor, predecessor[j]), j)
                without_j = _HMM_backtrack(take, predecessor, j - 1)
                key(picks) = [(candidates.start[rows[k]], candidates.stop[rows[k]]) for k in picks]
                choose = isless(key(with_j), key(without_j))
            end
        end
        take[j + 1] = choose
        utility[j + 1] = choose ? added : utility[j]
        counts[j + 1] = choose ? added_count : counts[j]
        total[j + 1] = choose ? added_total : total[j]
    end
    selected_rows = rows[_HMM_backtrack(take, predecessor, n)]
    intervals = DataFrame(candidates[selected_rows, :])
    intervals.utility = intervals.G .- penalty
    (; intervals, selected_rows, objective=sum(intervals.utility))
end

"""
    HMM_detect_intervals(residuals, reference; lengths, penalty, segments=...)

Scan and select anomalous intervals from normal forecast pseudo-residuals.
Return selected `intervals`, all `candidates`, their `selected_rows`, and the
penalized `objective`. Selected rows additionally contain whitened `mean`,
`covariance` (divisor `n_I`), and `regularized_covariance` for subsequent analysis.
For model-based calibration with a checked monitoring design, use the overload
accepting [`HMMIntervalCalibration`](@ref).
"""
function HMM_detect_intervals(residuals::AbstractMatrix, reference::HMMIntervalReference;
                              lengths, penalty::Real, segments=[1:size(residuals, 1)])
    candidates = HMM_scan_intervals(residuals, reference; lengths, segments)
    selection = HMM_select_intervals(candidates, penalty)
    x = _HMM_whiten_histories(residuals, reference, segments)
    summaries = [HMM_interval_score(@view x[(r.start + reference.q - 1):r.stop, :])
                 for r in eachrow(selection.intervals)]
    selection.intervals.mean = [s.mean for s in summaries]
    selection.intervals.covariance = [s.covariance for s in summaries]
    selection.intervals.regularized_covariance = [s.regularized_covariance for s in summaries]
    (; selection..., candidates)
end
