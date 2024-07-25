include("../pdgm_coag_icrp.jl")
include("../utils.jl")
using Serialization

α_true = 0.7
β_true = 0.2
θ_true = 100.0
m_true = 10
expected_edge_length_freqs = Dict(1 => 500.0, 2 => 7000.0, 3 => 1000.0, 7 => 15.0, 100 => 0.5, 1000 => 1.0)
In = simulate_ICRP(α_true, θ_true + m_true; expected_edge_length_freqs=expected_edge_length_freqs)
coag_In = pdgm_coagulate(In, α_true, β_true, θ_true, m_true)

println("$(length(In.nodes)) nodes, $(length(In.edge_lengths)) edges")
println("$(length(coag_In.nodes)) nodes, $(length(coag_In.edge_lengths)) edges")

params_true = Dict("α" => α_true, "β" => β_true, "θ" => θ_true, "m" => m_true)
serialize("synthetic.data", [In, coag_In, params_true])
