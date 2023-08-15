using Distributions, StatsPlots, ExpectationMaximization, JLD2, CSV, DataFrames

using Bessels, LogExpFunctions
# delta_radian
df = CSV.read("D:/CTHMM/FitHMM-jl-master/data/sample-data-clean.csv", DataFrame)
data = select(dropmissing(df),:delta_radian)[:,1]
sort!(data)

# A(k)=I1(k)/I0(k) is used to solve κ (measure of concentration)
# A(k) is symmetry about zero
# A(>=713.9) = NaN
function A(k) 
    A(k) = besseli(1,k)/besseli(0,k) 
    return A(k)
end  

# It is hard to calculate every x for each y or induce the inverse function of A(x) so here apply a distribution table (almost)
# let precision of κ be 0.01
invA_table = DataFrame([(0.00001:0.00001:0.99999) * 713.9, quantile(map(A,0:0.01:713.98),0.00001:0.00001:0.99999)], [:x, :y]) # 随后再调

# use the table to estimate
function invA(y)
    y = abs(y) # A(x) = -A(x)
    if y >= maximum(invA_table.y)
        x = 713.9
        elseif y <= maximum(invA_table.y) && y>= 0.979791 # filter(row -> row.x == 25, invA_table)[1,2] 
            x = mean(filter(row -> isapprox(row.y, y, atol = 0.00001) == true, invA_table).x) # linear interpolation
            else
                x = mean(filter(row -> isapprox(row.y, y, atol = 0.01) == true, invA_table).x)
            end
    return(round(x,digits=6))
end

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
        # α
        α[:] = mean(γ, dims = 1)
    
        # μ remain same to avoid NaN for neglected data around π

        # κ
        y = Vector{Float64}(undef,m)
        for j in 1:m
            y[j] = sum(γ[:,j] .* cos.(data .- μ[j])) / sum(γ[:,j])
        end 
        # use the table to estimate
        κ =  map(invA,y)

    return (p_iter = DataFrame(hcat(α,μ,κ),[:α, :μ, :κ]), ll = log(sum(exp.(LL .* γ))))
end

mix = MixtureModel(VonMises,[(-π/2, 20),(0.0, 20),(0.0, 2),(π/2, 200),(π, 20)], [0.05,0.44,0.44,0.05,0.02]) # μ fixed

# iteration
# I believe there are smarter way to iterate but it's ok if one of codes and I run well
atol = 1e-3
y = fit_MvM_fixed(mix,data).ll
# assign parameters to mix_new
tuples = []
for row in eachrow(hcat(fit_MvM_fixed(mix,data).p_iter.μ, fit_MvM_fixed(mix,data).p_iter.κ))
    push!(tuples, tuple(row...))
end
mix_new = MixtureModel(VonMises, tuples, fit_MvM_fixed(mix,data).p_iter.α)
# fit with mix_new
x = fit_MvM_fixed(mix_new,data).ll

iter = 1
while x - y >= atol
    # assign parameters to mix_new
    tuples = []
    for row in eachrow(hcat(fit_MvM_fixed(mix,data).p_iter.μ, fit_MvM_fixed(mix,data).p_iter.κ))
        push!(tuples, tuple(row...))
    end
    mix_new = MixtureModel(VonMises, tuples, fit_MvM_fixed(mix,data).p_iter.α)
    # fit with mix_new
    x = fit_MvM_fixed(mix_new,data).ll
    y = fit_MvM_fixed(mix,data).ll
    mix = mix_new
    print("iteration: $iter",", x = ",x,", y = ", y, ", x - y = ",x-y, "\n")
    iter += 1
end

# how to fit VonMises distribution with super large k (as large as 1000)? extend the invA_table or ...