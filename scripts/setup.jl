using DrWatson
@quickactivate "FitHMM-jl"

using Pkg
Pkg.add("Statistics")
Pkg.add("Dates")
Pkg.add("Pipe")
Pkg.add("Printf")
Pkg.add("Distributions")
Pkg.add("ShiftedArrays")
Pkg.add("GLM")
Pkg.add("CategoricalArrays")
Pkg.add("QuasiGLM")
Pkg.add("JLD2")
Pkg.add("StatsFuns")
Pkg.add("InvertedIndices")
Pkg.add("SpecialFunctions")
Pkg.add("QuadGK")
Pkg.add("Optim")
Pkg.add("Clustering")
Pkg.add("HypothesisTests")
Pkg.add("Roots")
Pkg.add("StatsBase")
Pkg.add("Bessels")
Pkg.add("Random")

include(srcdir("CTHMM.jl"))