import LogExpFunctions: logaddexp
import Distributions: Dirichlet, Multinomial

function pdgm_coag(
    nA::Vector{Int},
    α::Float64,
    β::Float64,
    θ::Float64,
    m::Int;
    return_info::Bool=true
)

    k = length(nA)
    nC = sample_counts_by_stick_breaking(β, θ, m)
    Km = length(nC)
    M = rand(Categorical(
            rand(Dirichlet(
                append!((nC .- β) ./ α, (θ + Km * β) / α)
            ))
        ), k)

    B0 = findall(M .== Km + 1)
    zB0, nB0 = sample_partition_by_crp(β / α, (θ + Km * β) / α, length(B0))
    kB0 = length(nB0)

    zcA = zeros(Int, k)
    ncA0 = zeros(Int, kB0)
    for (l, i) in enumerate(B0)
        zcA[i] = zB0[l]
        ncA0[zB0[l]] += nA[i]
    end

    ncA = zeros(Int, 0)
    nB = zeros(Int, 0)
    l = kB0 + 1
    for j in 1:Km
        Bj = findall(M .== j)
        nBj = length(Bj)
        if nBj > 0
            ncA = append!(ncA, sum(nA[Bj]))
            nB = append!(nB, nBj)
            zcA[Bj] .= l
            l += 1
        end
    end

    if return_info
        return zcA, ncA, ncA0, nB, nB0, nC
    else
        return zcA, vcat(ncA, ncA0)
    end

end

function log_pdgm_coag(
    nB::Vector{Int},
    nB0::Vector{Int},
    nC::Vector{Int},
    α::Float64,
    β::Float64,
    θ::Float64,
    m::Int
)
    lj = log_eppf(nC, β, θ)
    k = sum(nB) + sum(nB0)
    Km = length(nC)

    base = (θ + m) / α
    lj += loggamma(base) - loggamma(base + k)
    base = (θ + Km * β) / α
    lj += loggamma(base + sum(nB0)) - loggamma(base)
    for j = 1:Km
        if nC[j] > 0
            base = (nC[j] - β) / α
            lj += loggamma(base + nB[j]) - loggamma(base)
        end
    end

    lj += log_eppf(nB0, β / α, (θ + Km * β) / α)

    return lj
end


function estimate_log_pdgm_coag(
    ncA::Vector{Int},
    ntB::Vector{Int},
    α::Float64,
    β::Float64,
    θ::Float64,
    m::Int;
    num_samples::Int=10
)

    kp = length(ncA)
    n = sum(ncA)

    lj = -Inf
    for _ in 1:num_samples
        nC = zeros(Int, 0)
        nB0 = zeros(Int, 0)
        nB = zeros(Int, 0)

        N = rand(Multinomial(m, rand(Dirichlet(append!(ncA .- β, θ + kp * β)))))
        log_q = loggamma(θ + n) - loggamma(θ + n + m)
        
        for j in 1:kp
            if N[j] > 0
                base = ncA[j] - β
                log_q += loggamma(base + N[j]) - loggamma(base)
                nC = append!(nC, N[j])
                nB = append!(nB, ntB[j])
            else
                nB0 = append!(nB0, ntB[j])
            end
        end
        base = θ + kp * β
        log_q += loggamma(base + N[kp+1]) - loggamma(base)

        log_p = log_pdgm_coag(nB, nB0, nC, α, β, θ, m)
        lj = logaddexp(lj, log_p - log_q)
    end

    lj = lj - log(num_samples)
    return lj
end