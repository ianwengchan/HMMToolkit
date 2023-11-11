# delta_radian exploration

using DrWatson
@quickactivate "FitHMM-jl"

using Distributions, StatsPlots, JLD2, CSV, DataFrames
using LogExpFunctions
using Bessels

# load data
df_longer = load(datadir("df_longer.jld2"), "df_longer")
delta_radian = select(dropmissing(df_longer), :delta_radian).delta_radian # missings

# take a look
density(delta_radian)

# function A(x)
function A(x)
    y = besseli1(x) / besseli0(x)
    return y
end

# It is hard to calculate every x for each y or induce the inverse function of A(x) so here apply a distribution table
# if κ -> 1 VonMises distribution is already uniform so possible κ value below 1 is dropped
invA_table = DataFrame([1:0.1:713.98, map(A,1:0.1:713.98)], [:x, :y])

# use the table to estimate κ
function invA(y)
    y = abs(y) # A(x) = -A(x)
    if y >= maximum(invA_table.y)
        x = 713.9 + (y - maximum(invA_table.y)) / 0.000005 # a problem!!!!!!
        elseif y <= maximum(invA_table.y) && y>= 0.979791 # filter(row -> row.x == 25, invA_table)[1,2] 
            x = mean(filter(row -> isapprox(row.y, y, atol = 0.001) == true, invA_table).x) # linear interpolation
            else
                x = mean(filter(row -> isapprox(row.y, y, atol = 0.01) == true, invA_table).x)
            end
    return(round(x,digits=6))
end

# fit_MvM_fixed
# fixed μ for preventing overfitted central values 
function fit_MvM_fixed(mix::MixtureModel, data::AbstractVecOrMat)
    m = copy(ncomponents(mix)) # number of components, set manually
    dists = copy(components(mix)) # single VonMises(vM) distribution
    n = copy(length(data)) # number of observations 
    
    # parameters to be estimated (initial)
    α = copy(probs(mix)) 
    μ = copy(DataFrame(dists).μ)   
    κ = copy(DataFrame(dists).κ)

    # Allocate memory
    LL = zeros(n, m)
    γ = similar(LL)
    c = zeros(n)
    
        # E step
        # log likelihood matrix
        for j in eachindex(dists)
            LL[:, j] .= log(α[j]) .+ logpdf.(dists[j], data)
        end
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    
        # γ_ij
        logsumexp!(c, LL)
        γ[:, :] .= exp.(LL .- c)
    
        # M_step
        # update parameters
        # α
        α[:] = mean(γ, dims = 1)
    
        # μ remain same to avoid NaN for data value around π

        # κ
        y = Vector{Float64}(undef,m)
        for j in 1:m
            y[j] = sum(γ[:,j] .* cos.(data .- μ[j])) / sum(γ[:,j])
        end 
        # use the table to estimate
        κ =  map(invA,y)
        
        # organize output
        tuples = []
        for row in eachrow(hcat(μ, κ))
            push!(tuples, tuple(row...))
        end
        mix = MixtureModel(VonMises, tuples, α)
        
        # log likelihood
        ll = sum(c)

    return (mix = mix, ll = ll)
end

# iteration
atol = 1e-3
logtot = fit_MvM_fixed(mix,data).ll
logtotp = fit_MvM_fixed(fit_MvM_fixed(mix,data).mix,data).ll
iter = 0

while abs(logtotp - logtot) >= atol
    iter += 1
    mix = fit_MvM_fixed(mix,data).mix
    logtot = fit_MvM_fixed(mix,data).ll
    logtotp = fit_MvM_fixed(fit_MvM_fixed(mix,data).mix,data).ll
    println("iteration = $iter, logtotp = $logtotp, logtot = $logtot, diff = $(abs(logtotp - logtot))")
end

