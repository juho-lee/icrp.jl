include("../pdgm_coag_icrp.jl")
include("../utils.jl")
using Serialization

α = 0.7
β = 0.2
θ = 100.0
m = 10
expected_edge_length_freqs = Dict(1 => 500.0, 2 => 7000.0, 3 => 1000.0, 7 => 15.0, 100 => 0.5, 1000 => 1.0)
In = simulate_ICRP(α, θ + m; expected_edge_length_freqs=expected_edge_length_freqs)
coag_In = pdgm_coagulate(In, α, β, θ, m)

println("$(length(In.nodes)) nodes, $(length(In.edge_lengths)) edges")
println("$(length(coag_In.nodes)) nodes, $(length(coag_In.edge_lengths)) edges")

params_true = Dict{String, Float64}("α" => α, "β" => β, "θ" => θ, "m" => m)
serialize("synthetic.data", [In, coag_In, params_true])
