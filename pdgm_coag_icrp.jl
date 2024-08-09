include("icrp.jl")
include("pdgm.jl")
using ProgressBars

function pdgm_coagulate(In::Interactions, α::Float64, β::Float64, θ::Float64, m::Int)

    partition = get_partition(In)
    coag_edge_sequence = deepcopy(In.edge_sequence)

    zcA, ncA = pdgm_coag(In.degrees, α, β, θ, m; return_info=false)

    coag_nodes = Vector{Node}(undef, length(ncA))
    for j in eachindex(ncA)
        coag_nodes[j] = Node(j)
    end

    for (i, z) in enumerate(zcA)
        coag_edge_sequence[partition[i]] .= z
        In.nodes[i].parent = coag_nodes[z]
        push!(coag_nodes[z].children, In.nodes[i])
    end

    coag_degrees = zeros(Int, length(coag_nodes))
    for i in coag_edge_sequence
        coag_degrees[i] += 1
    end

    return Interactions(
        coag_nodes,
        coag_edge_sequence,
        deepcopy(In.edge_lengths);
        degrees=coag_degrees
    )
end

function pdgm_coag_log_likelihood(
    coag_In::Interactions,
    α::Float64, β::Float64, θ::Float64, m::Int;
    num_samples::Int=10
)
    return estimate_log_pdgm_coag(
        coag_In.degrees,
        [length(node.children) for node in coag_In.nodes],
        α, β, θ, m; num_samples=num_samples)
end

# I ~ ICRP(α, θ + m, n)
# coag_I ~ ICRP(β, θ, n)
mutable struct PDGMCoagICRPSampler
    In::Interactions
    coag_In::Interactions

    α_rwmh::RandomWalkMetropolisHastings
    β_over_α_rwmh::RandomWalkMetropolisHastings
    θm_rwmh::RandomWalkMetropolisHastings
    κ_rwmh::RandomWalkMetropolisHastings

    t::Float64
    log_ν::Dict{Int,Float64}


    function PDGMCoagICRPSampler(In::Interactions, coag_In::Interactions)

        icrp_sampler = ICRPSampler(In)
        α_rwmh = icrp_sampler.α_rwmh
        t = icrp_sampler.t
        log_ν = icrp_sampler.log_ν


        β, θ = pyp_rwmh(coag_In.degrees, 1000)
        β_over_α = β / get(α_rwmh)
        β_over_α_rwmh = RandomWalkMetropolisHastings(SigmoidBijection())
        initialize!(β_over_α_rwmh, β_over_α)

        θm_rwmh = icrp_sampler.θ_rwmh
        κ_rwmh = RandomWalkMetropolisHastings(SigmoidBijection())
        κ_init = max(min(θ / get(θm_rwmh), 0.9), 0.1)
        initialize!(κ_rwmh, κ_init)

        return new(In, coag_In, α_rwmh, β_over_α_rwmh, θm_rwmh, κ_rwmh, t, log_ν)
    end
end

function Base.getproperty(obj::PDGMCoagICRPSampler, sym::Symbol)
    if sym === :α
        return get(Base.getfield(obj, :α_rwmh))
    elseif sym === :β_over_α
        return get(Base.getfield(obj, :β_over_α_rwmh))
    elseif sym === :β
        return get(Base.getfield(obj, :α_rwmh)) * get(Base.getfield(obj, :β_over_α_rwmh))
    elseif sym === :θm
        return get(Base.getfield(obj, :θm_rwmh))
    elseif sym === :κ
        return get(Base.getfield(obj, :κ_rwmh))
    elseif sym === :θ
        θm = get(Base.getfield(obj, :θm_rwmh))
        κ = get(Base.getfield(obj, :κ_rwmh))
        return θm * κ
    elseif sym === :m
        θm = get(Base.getfield(obj, :θm_rwmh))
        κ = get(Base.getfield(obj, :κ_rwmh))
        m_real = θm * (1.0 - κ)
        return Int(ceil(m_real))
    else
        return Base.getfield(obj, sym)
    end
end

function log_likel(sampler::PDGMCoagICRPSampler)
    return (
        icrp_log_likel(sampler.In, sampler.α, sampler.θm, sampler.t, sampler.log_ν) +
        pdgm_coag_log_likelihood(sampler.coag_In, sampler.α, sampler.β, sampler.θ, sampler.m)
    )
end

function sample_α!(sampler::PDGMCoagICRPSampler)
    k = length(sampler.In.nodes)

    function log_likel(α::Float64)
        β = sampler.β_over_α * α
        return (
            k * log(α) + loggamma(sampler.θm / α + k) - loggamma(sampler.θm / α) +
            sum(loggamma.(sampler.In.degrees .- α) .- loggamma(1 - α)) +
            pdgm_coag_log_likelihood(sampler.coag_In, α, β, sampler.θ, sampler.m)
        )
    end
    mh_step!(sampler.α_rwmh, log_likel)
end

function sample_θm!(sampler::PDGMCoagICRPSampler)
    k = length(sampler.In.nodes)
    n = sum(sampler.In.edge_lengths)
    log_t = log(sampler.t)
    α = sampler.α
    κ = sampler.κ

    function log_likel(θm::Float64)
        θ = θm * κ
        m = Int(ceil(θm * (1.0 - κ)))
        return (
            θm * log_t + loggamma(θm / α + k) - loggamma(θm / α) - loggamma(θm + n) +
            pdgm_coag_log_likelihood(sampler.coag_In, α, sampler.β, θ, m)
        )
    end

    mh_step!(sampler.θm_rwmh, log_likel)
end

function sample_β_over_α!(sampler::PDGMCoagICRPSampler)

    function log_likel(β_over_α::Float64)
        return pdgm_coag_log_likelihood(
            sampler.coag_In, sampler.α, sampler.α * β_over_α, sampler.θ, sampler.m
        )
    end

    mh_step!(sampler.β_over_α_rwmh, log_likel)
end

function sample_κ!(sampler::PDGMCoagICRPSampler)

    k = length(sampler.In.nodes)
    n = sum(sampler.In.edge_lengths)
    α = sampler.α
    θm = sampler.θm
    log_t = log(sampler.t)

    function log_likel(κ::Float64)
        θ = θm * κ
        m = Int(ceil(θm * (1 - κ)))
        return (
            θm * log_t + loggamma(θm / α + k) - loggamma(θm / α) - loggamma(θm + n) +
            pdgm_coag_log_likelihood(sampler.coag_In, sampler.α, sampler.β, θ, m)
        )

    end

    mh_step!(sampler.κ_rwmh, log_likel)
end

function sample_t!(sampler::PDGMCoagICRPSampler)
    sampler.t = rand(Gamma(sampler.θ + sampler.m))
    log_t = log(sampler.t)
    for (c, nc) in sampler.In.edge_length_freqs
        sampler.log_ν[c] = log(nc) - c * log_t
    end
end


function step!(sampler::PDGMCoagICRPSampler)
    sample_α!(sampler)
    sample_θm!(sampler)
    sample_β_over_α!(sampler)
    sample_κ!(sampler)
    sample_t!(sampler)
end


mutable struct PDGMCoagICRPChain
    α::Vector
    β::Vector
    θ::Vector
    m::Vector{Int}
    t::Vector
    log_ν::Vector{Dict{Int,Float64}}


    function PDGMCoagICRPChain()
        return new(zeros(0), zeros(0), zeros(0),
            zeros(Int, 0), zeros(0),
            Vector{Dict{Int,Float64}}(undef, 0))
    end
end


function run_sampler(In::Interactions, coag_In::Interactions, num_steps::Int;
    burn_in::Int=-1, thin::Int=10, print_every::Int=1000)

    if burn_in == -1
        burn_in = convert(Int, trunc(num_steps / 2))
    end

    sampler = PDGMCoagICRPSampler(In, coag_In)
    chain = PDGMCoagICRPChain()

    for i in 1:num_steps
        step!(sampler)

        if i % print_every == 0
            ll = log_likel(sampler)
            line = (
                f"step {i} ll {ll:.4e} " *
                f"α {sampler.α:.4f} ({acc_rate(sampler.α_rwmh):.4f}) " *
                f"β {sampler.β:.4f} ({acc_rate(sampler.β_over_α_rwmh):.4f}) " *
                f"θ {sampler.θ:.4f} ({acc_rate(sampler.θm_rwmh):.4f}) " *
                f"m {sampler.m} ({acc_rate(sampler.κ_rwmh):.4f}) " *
                f"t {sampler.t:.4f}")
            println(line)
        end

        if i > burn_in && i % thin == 0
            push!(chain.α, sampler.α)
            push!(chain.β, sampler.β)
            push!(chain.θ, sampler.θ)
            push!(chain.m, sampler.m)
            push!(chain.t, sampler.t)
            push!(chain.log_ν, deepcopy(sampler.log_ν))
        end
    end

    return chain
end

mutable struct PDGMCoagICRPPred
    num_nodes::Vector{Int}
    num_edges::Vector{Int}
    degrees::Vector{Vector{Int}}
    degree_kss::Vector

    coag_num_nodes::Vector{Int}
    coag_degrees::Vector{Vector{Int}}
    coag_degree_kss::Vector

    parents::Vector{Vector{Int}}
    num_children::Vector{Vector{Int}}


    function PDGMCoagICRPPred()
        return new(
            zeros(Int, 0), zeros(Int, 0), Vector{Vector{Int}}(undef, 0), zeros(0),
            zeros(Int, 0), Vector{Vector{Int}}(undef, 0), zeros(0),
            Vector{Vector{Int}}(undef, 0),
            Vector{Vector{Int}}(undef, 0)
        )
    end

end

function simulate_predictives(In::Interactions, coag_In::Interactions,
    chain::PDGMCoagICRPChain; thin::Int=5)


    αs = chain.α[1:thin:end]
    βs = chain.β[1:thin:end]
    θs = chain.θ[1:thin:end]
    ms = chain.m[1:thin:end]
    ts = chain.t[1:thin:end]
    log_νs = chain.log_ν[1:thin:end]
    pred = PDGMCoagICRPPred()

    for (α, β, θ, m, t, log_ν) in ProgressBar(zip(αs, βs, θs, ms, ts, log_νs))
        log_t = log(t)
        n, M = 0, 0
        for (c, log_νc) in log_ν
            nc = rand(Poisson(exp(log_νc + c * log_t)))
            n += nc
            M += c * nc
        end
        degrees = sample_counts_by_stick_breaking(α, θ + m, M)
        push!(pred.num_nodes, length(degrees))
        push!(pred.num_edges, n)
        push!(pred.degrees, degrees)
        push!(pred.degree_kss, ApproximateTwoSampleKSTest(In.degrees, degrees).δ)

        zcA, ncA, ncA0, nB, nB0, _ = pdgm_coag(degrees, α, β, θ, m)
        ntcA = vcat(ncA, ncA0)
        ntB = vcat(nB, nB0)

        push!(pred.parents, zcA)
        push!(pred.coag_degrees, ntcA)
        push!(pred.num_children, ntB)

        push!(pred.coag_num_nodes, length(ntcA))
        push!(pred.coag_degree_kss, ApproximateTwoSampleKSTest(coag_In.degrees, ntcA).δ)
    end

    return pred
end