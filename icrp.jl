using SpecialFunctions: loggamma
using LogExpFunctions: logsumexp, logaddexp, logistic, logit
using Random: shuffle!
import HypothesisTests: ApproximateTwoSampleKSTest
using ProgressBars

include("interactions.jl")
include("pyp.jl")

function simulate_ICRP(
    α::Float64, θ::Float64;
    expected_edge_length_freqs::Union{Nothing,Dict{Int,Float64}}=nothing
)

    if expected_edge_length_freqs === nothing
        t = rand(Gamma(θ))
        expected_edge_length_freqs = Dict(2 => t^2)
    end

    edge_lengths = zeros(Int, 0)
    edge_length_freqs = Dict{Int,Int}()
    for (c, enc) in expected_edge_length_freqs
        nc = rand(Poisson(enc))
        if nc > 0
            edge_length_freqs[c] = nc
            append!(edge_lengths, fill(c, nc))
        end
    end

    shuffle!(edge_lengths)

    edge_sequence, degrees = sample_partition_by_crp(α, θ, sum(edge_lengths))
    nodes = [Node(j) for j in eachindex(degrees)]

    return Interactions(nodes, edge_sequence, edge_lengths;
        edge_length_freqs=edge_length_freqs, degrees=degrees)
end

mutable struct ICRPSampler
    In::Interactions
    α_rwmh::RandomWalkMetropolisHastings
    θ_rwmh::RandomWalkMetropolisHastings
    t::Float64
    log_ν::Dict{Int,Float64}

    function ICRPSampler(In::Interactions)

        α, θ = pyp_rwmh(In.degrees, 1000)
        α_rwmh = RandomWalkMetropolisHastings(SigmoidBijection())
        initialize!(α_rwmh, α)
        θ_rwmh = RandomWalkMetropolisHastings(ExpBijection())
        initialize!(θ_rwmh, θ)
        t = rand(Gamma(θ))

        log_ν = Dict{Int,Float64}()
        log_t = log(t)
        for (c, nc) in In.edge_length_freqs
            log_ν[c] = log(nc) - c * log_t
        end
        return new(In, α_rwmh, θ_rwmh, t, log_ν)
    end
end

function Base.getproperty(obj::ICRPSampler, sym::Symbol)
    if sym === :α
        return get(Base.getfield(obj, :α_rwmh))
    elseif sym === :θ
        return get(Base.getfield(obj, :θ_rwmh))
    else
        return Base.getfield(obj, sym)
    end
end

function icrp_log_likel(In::Interactions, α::Float64, θ::Float64,
    t::Float64, log_ν::Dict{Int,Float64})
    m = sum(In.edge_lengths)
    log_t = log(t)
    ll = (θ - 1) * log_t - t - loggamma(θ)
    ll += log_eppf(In.degrees, α, θ; n=m, k=length(In.degrees))
    return ll
end

function log_likel(sampler::ICRPSampler)
    return icrp_log_likel(sampler.In, sampler.α, sampler.θ, sampler.t, sampler.log_ν)
end

function sample_α!(sampler::ICRPSampler)
    k = length(sampler.In.nodes)
    θ = sampler.θ
    function log_f(α::Float64)
        return (
            k * log(α) + loggamma(θ / α + k) - loggamma(θ / α) +
            sum(loggamma.(sampler.In.degrees .- α) .- loggamma(1 - α))
        )
    end
    mh_step!(sampler.α_rwmh, log_f)
end

function sample_θ!(sampler::ICRPSampler)
    k = length(sampler.In.nodes)
    m = sum(sampler.In.edge_lengths)
    log_t = log(sampler.t)
    α = sampler.α
    function log_f(θ::Float64)
        return θ * log_t + loggamma(θ / α + k) - loggamma(θ / α) - loggamma(θ + m)
    end
    mh_step!(sampler.θ_rwmh, log_f)
end

function sample_t!(sampler::ICRPSampler)
    sampler.t = rand(Gamma(sampler.θ))
    log_t = log(sampler.t)
    for (c, nc) in sampler.In.edge_length_freqs
        sampler.log_ν[c] = log(nc) - c * log_t
    end
end


function step!(sampler::ICRPSampler)
    sample_α!(sampler)
    sample_θ!(sampler)
    sample_t!(sampler)
end

mutable struct ICRPChain
    α::Vector
    θ::Vector
    t::Vector
    log_ν::Vector{Dict{Int,Float64}}

    function ICRPChain()
        return new(zeros(0), zeros(0), zeros(0), Vector{Dict{Int,Float64}}(undef, 0))
    end

end

function run_sampler(In::Interactions, num_steps::Int;
    burn_in::Int=-1, thin::Int=10, print_every::Int=1000)

    if burn_in == -1
        burn_in = convert(Int, trunc(num_steps / 2))
    end

    sampler = ICRPSampler(In)
    chain = ICRPChain()

    for i in 1:num_steps
        step!(sampler)


        if i % print_every == 0
            ll = log_likel(sampler)
            line = (
                f"step {i} ll {ll:.4e} " *
                f"α {sampler.α:.4f} ({acc_rate(sampler.α_rwmh):.4f}) " *
                f"θ {sampler.θ:.4f} ({acc_rate(sampler.θ_rwmh):.4f}) " *
                f"t {sampler.t:.4f}")
            println(line)
        end

        if i > burn_in && i % thin == 0
            push!(chain.α, sampler.α)
            push!(chain.θ, sampler.θ)
            push!(chain.t, sampler.t)
            push!(chain.log_ν, deepcopy(sampler.log_ν))
        end
    end

    return chain
end

mutable struct ICRPPred
    num_nodes::Vector{Int}
    num_edges::Vector{Int}
    degrees::Vector{Vector{Int}}
    degree_kss::Vector

    function ICRPPred()
        return new(
            zeros(Int, 0),
            zeros(Int, 0),
            Vector{Vector{Int}}(undef, 0),
            zeros(0)
        )
    end

end

function simulate_predictives(In::Interactions, chain::ICRPChain; thin::Int=5)
    αs = chain.α[1:thin:end]
    θs = chain.θ[1:thin:end]
    ts = chain.t[1:thin:end]
    log_νs = chain.log_ν[1:thin:end]
    pred = ICRPPred()

    for (α, θ, t, log_ν) in ProgressBar(zip(αs, θs, ts, log_νs))
        log_t = log(t)
        n, m = 0, 0
        for (c, log_νc) in log_ν
            nc = rand(Poisson(exp(log_νc + c * log_t)))
            n += nc
            m += c * nc
        end
        degrees = sample_counts_by_stick_breaking(α, θ, m)
        push!(pred.num_nodes, length(degrees))
        push!(pred.num_edges, n)
        push!(pred.degrees, degrees)
        push!(pred.degree_kss, ApproximateTwoSampleKSTest(In.degrees, degrees).δ)
    end

    return pred
end