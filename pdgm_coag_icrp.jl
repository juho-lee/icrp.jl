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
    θ_rwmh::RandomWalkMetropolisHastings
    m_rwmh::RandomWalkMetropolisHastings

    t::Float64
    log_ν::Dict{Int,Float64}


    function PDGMCoagICRPSampler(In::Interactions, coag_In::Interactions;
        m_est::Union{Float64,Nothing}=nothing)

        icrp_sampler = ICRPSampler(In)
        α_rwmh = icrp_sampler.α_rwmh
        t = icrp_sampler.t
        log_ν = icrp_sampler.log_ν


        β, θ = pyp_rwmh(coag_In.degrees, 1000)
        β_over_α = β / get(α_rwmh)
        β_over_α_rwmh = RandomWalkMetropolisHastings(SigmoidBijection())
        initialize!(β_over_α_rwmh, β_over_α)

        if m_est === nothing
            m_est = max(icrp_sampler.θ - θ, 1)
        end

        m_lb = 1
        m_ub = 4.0 * m_est

        m_rwmh = RandomWalkMetropolisHastings(
            CompositeBijection(SigmoidBijection(m_ub), TranslateBijection(m_lb));
        )
        initialize!(m_rwmh, m_est)
        θ_rwmh = RandomWalkMetropolisHastings(ExpBijection())
        initialize!(θ_rwmh, max(θ, 0.1))

        return new(In, coag_In, α_rwmh, β_over_α_rwmh, θ_rwmh, m_rwmh, t, log_ν)
    end
end

function Base.getproperty(obj::PDGMCoagICRPSampler, sym::Symbol)
    if sym === :α
        return get(Base.getfield(obj, :α_rwmh))
    elseif sym === :β_over_α
        return get(Base.getfield(obj, :β_over_α_rwmh))
    elseif sym === :β
        return get(Base.getfield(obj, :α_rwmh)) * get(Base.getfield(obj, :β_over_α_rwmh))
    elseif sym === :θ
        return get(Base.getfield(obj, :θ_rwmh))
    elseif sym === :m
        m_real = get(Base.getfield(obj, :m_rwmh))
        return Int(ceil(m_real))
    else
        return Base.getfield(obj, sym)
    end
end

function log_likel(sampler::PDGMCoagICRPSampler)
    α, β, θ, m = sampler.α, sampler.β, sampler.θ, sampler.m
    org_ll = icrp_log_likel(sampler.In, α, θ + m, sampler.t, sampler.log_ν)
    coag_ll = pdgm_coag_log_likelihood(sampler.coag_In, α, β, θ, m)
    return org_ll + coag_ll
end

function sample_α!(sampler::PDGMCoagICRPSampler)
    k = length(sampler.In.nodes)
    θm = sampler.θ + sampler.m

    function log_likel(α::Float64)
        β = sampler.β_over_α * α
        return (
            k * log(α) + loggamma(θm / α + k) - loggamma(θm / α) +
            sum(loggamma.(sampler.In.degrees .- α) .- loggamma(1 - α)) +
            pdgm_coag_log_likelihood(sampler.coag_In, α, β, sampler.θ, sampler.m)
        )
    end
    mh_step!(sampler.α_rwmh, log_likel)
end

function sample_θ!(sampler::PDGMCoagICRPSampler)
    k = length(sampler.In.nodes)
    n = sum(sampler.In.edge_lengths)
    log_t = log(sampler.t)
    α = sampler.α
    m = sampler.m

    function log_likel(θ::Float64)
        θm = θ + m
        return (
            θm * log_t + loggamma(θm / α + k) - loggamma(θm / α) - loggamma(θm + n) +
            pdgm_coag_log_likelihood(sampler.coag_In, α, sampler.β, θ, m)
        )
    end

    mh_step!(sampler.θ_rwmh, log_likel)
end

function sample_β_over_α!(sampler::PDGMCoagICRPSampler)

    function log_likel(β_over_α::Float64)
        return pdgm_coag_log_likelihood(
            sampler.coag_In, sampler.α, sampler.α * β_over_α, sampler.θ, sampler.m
        )
    end

    mh_step!(sampler.β_over_α_rwmh, log_likel)
end

function sample_m!(sampler::PDGMCoagICRPSampler)

    k = length(sampler.In.nodes)
    M = sum(sampler.In.edge_lengths)
    α = sampler.α
    log_t = log(sampler.t)

    function log_likel(m_real::Float64)
        θm = sampler.θ + Int(ceil(m_real))
        return (
            θm * log_t + loggamma(θm / α + k) - loggamma(θm / α) - loggamma(θm + M) +
            pdgm_coag_log_likelihood(sampler.coag_In,
                sampler.α, sampler.β, sampler.θ, Int(ceil(m_real)))
        )

    end

    mh_step!(sampler.m_rwmh, log_likel)
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
    sample_θ!(sampler)
    sample_β_over_α!(sampler)
    sample_m!(sampler)
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
    burn_in::Int=-1, thin::Int=10, print_every::Int=1000, m_est::Union{Float64,Nothing}=nothing)

    if burn_in == -1
        burn_in = convert(Int, trunc(num_steps / 2))
    end

    sampler = PDGMCoagICRPSampler(In, coag_In; m_est=m_est)
    chain = PDGMCoagICRPChain()

    for i in 1:num_steps
        step!(sampler)

        if i % print_every == 0
            ll = log_likel(sampler)
            line = (
                f"step {i} ll {ll:.4e} " *
                f"α {sampler.α:.4f} ({acc_rate(sampler.α_rwmh):.4f}) " *
                f"β {sampler.β:.4f} ({acc_rate(sampler.β_over_α_rwmh):.4f}) " *
                f"θ {sampler.θ:.4f} ({acc_rate(sampler.θ_rwmh):.4f}) " *
                f"m {sampler.m} ({acc_rate(sampler.m_rwmh):.4f}) " *
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

# function simulate_predictives(In::Interactions, coag_In::Interactions,
#     chain::PDGMCoagICRPChain; thin::Int=5)


#     αs = chain.α[1:thin:end]
#     βs = chain.β[1:thin:end]
#     θs = chain.θ[1:thin:end]
#     ms = chain.m[1:thin:end]
#     ts = chain.t[1:thin:end]
#     log_νs = chain.log_ν[1:thin:end]
#     pred = PDGMCoagICRPPred()

#     for (α, β, θ, m, t, log_ν) in ProgressBar(zip(αs, βs, θs, ms, ts, log_νs))
#         log_t = log(t)
#         n, M = 0, 0
#         for (c, log_νc) in log_ν
#             nc = rand(Poisson(exp(log_νc + c * log_t)))
#             n += nc
#             M += c * nc
#         end
#         degrees = sample_counts_by_stick_breaking(α, θ + m, M)
#         push!(pred.num_nodes, length(degrees))
#         push!(pred.num_edges, n)
#         push!(pred.degrees, degrees)
#         push!(pred.degree_kss, ApproximateTwoSampleKSTest(In.degrees, degrees).δ)

#         zB, nB0, nBj, nC, nS0t, nSj = sample_pdgm_coag(degrees, α, β, θ, m)
#         nB = vcat(nB0, nBj)
#         nS = vcat(nS0t, nSj)

#         # zB, nB = sample_pdgm_coag(degrees, α, β, θ, m; return_info=false)
#         # coag_degrees = zeros(Int, length(nB))
#         # for (i, z) in enumerate(zB)
#         #     coag_degrees[z] += degrees[i]
#         # end

#         push!(pred.parents, zB)
#         push!(pred.coag_degrees, nB)
#         push!(pred.num_children, nS)
#         # push!(pred.nB, nB)

#         push!(pred.coag_num_nodes, length(nB))
#         # push!(pred.coag_degrees, coag_degrees)
#         push!(pred.coag_degree_kss, ApproximateTwoSampleKSTest(coag_In.degrees, nB).δ)
#     end

#     return pred
# end