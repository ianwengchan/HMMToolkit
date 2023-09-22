using DrWatson
@quickactivate "FitHMM-jl"

using Base.Threads

println("Hello World!")

@threads for i in 1:100
    println("Hello World from Thread $i.")
end