# Run from any working directory with:
# julia --project=/path/to/HMMToolkit /path/to/HMMToolkit/examples/anomalous-intervals.jl
# These are illustrative synthetic parameters; no model fitting or data files.
ENV["HMMTOOLKIT_SKIP_PLOTS"] = get(ENV, "HMMTOOLKIT_SKIP_PLOTS", "true")
using Random, DataFrames
if !isdefined(Main, :HMMToolkit)
    include(joinpath(@__DIR__, "..", "src", "HMMToolkit.jl"))
end
using .HMMToolkit

Random.seed!(406)
responses = [:channel1, :channel2]
transition = [0.95 0.05; 0.10 0.90]
initial = [2/3, 1/3]
states = [NormalExpert(-0.3, 0.8) NormalExpert(0.6, 1.0);
          NormalExpert(0.2, 1.0) NormalExpert(-0.5, 0.8)]
design = HMMIntervalDesign(100)
lengths = [8, 12, 16]

# Common reference and a separate whole-search calibration bank.
# Small banks keep this example quick; the API defaults are 500 and 1999.
bank = HMM_interval_reference([design], responses, transition, initial, states;
    q=1, simulations=30)
calibration = HMM_calibrate_interval_penalty(design, responses, transition,
    initial, states, bank.reference; lengths, simulations=99)

trip = DTHMM_sim_time_series(1, transition, initial, states, design; response_list=responses)
trip[40:51, :channel1] .+= 3.0
trip[40:51, :channel2] .+= 3.0
prepared = HMM_interval_residuals(trip, responses, transition, initial, states, design)
detected = HMM_detect_intervals(prepared.residuals, calibration)

println("Calibrated penalty: ", calibration.penalty)
println("Selected objective: ", detected.objective)
show(stdout, MIME("text/plain"), select(detected.intervals,
    :observation_start, :observation_stop, :G, :G_mean, :G_cov, :utility, :pvalue))
println()
