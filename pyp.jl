import Distributions: Beta, Poisson, Gamma, Categorical
import SpecialFunctions: loggamma, digamma
import LogExpFunctions: log1mexp, logistic, logit
using PyFormattedStrings
include("rwmh/rwmh.jl")

function sample_partition_by_crp(α::Float64, θ::Float64, n::Int)
    nj = Vector{Int}()
    indices = zeros(Int, n)
    for i in 1:n
        k = length(nj)
        prob = append!(copy(nj) .- α, θ + α * k)
        prob = prob / sum(prob)
        j = rand(Categorical(prob))
        indices[i] = j
        if j == k + 1
            nj = append!(nj, 1)
        else
            nj[j] += 1
        end
    end
    return indices, nj
end


function sample_counts_by_stick_breaking(α::Float64, θ::Float64, n::Int;
    thres=1.0e-6, max_iter=1.0e+7, min_n=1000)

    # Just use CRP if n < min_n
    if n <= min_n
        _, nj = sample_partition_by_crp(α, θ, n)
        return nj
    end

    log_1mb_accum = 0
    remainder = n
    cnts = zeros(Int, 0)

    for j in 1:max_iter
        log_b = log(rand(Beta(1 - α, θ + j * α)))
        log_w = log_b + log_1mb_accum
        log_1mb_accum += log1mexp(log_b)

        cnt = rand(Poisson(n * exp(log_w)))

        if remainder < cnt
            break
        elseif cnt > 0
            remainder -= cnt
            push!(cnts, cnt)
        end

        if remainder / n <= thres
            break
        end

    end

    if remainder > 0
        cnts = append!(cnts, ones(Int, remainder))
    end
    return cnts
end

function asymp_pj(j::Int, α::Float64)
    if α > 0
        return exp.(
            log(α) + loggamma(j - α) - loggamma(1 - α) - loggamma(j + 1)
        )
    else
        return 0
    end
end

function log_eppf_const(n::Int, k::Int, α::Float64, θ::Float64)
    if α == 0
        return k * log(θ) + loggamma(θ) - loggamma(θ + n)
    else
        if θ == 0
            return (k - 1) * log(α) + loggamma(k) - loggamma(n)
        else
            # return (k - 1) * log(α) + loggamma(θ / α + k) - loggamma(θ / α + 1) + loggamma(θ + 1) - loggamma(θ + n)
            return sum(log.(θ .+ α .* (1:k-1))) + loggamma(θ + 1) - loggamma(θ + n)
        end
    end
end

function log_eppf(nj::Vector{Int}, α::Float64, θ::Float64; n=nothing, k=nothing)
    if n === nothing
        n = sum(nj)
    end
    if k === nothing
        k = length(nj)
    end
    return log_eppf_const(n, k, α, θ) + sum(loggamma.(nj .- α) .- loggamma(1 - α))
end

function pyp_mle(nj::Vector{Int}; η=1.0e-4, max_iter=10000, thres=1.0e-4, verbose=false)

    n, k = sum(nj), length(nj)

    log_θ = randn()
    θ = exp(log_θ)
    logit_α = randn()
    α = logistic(logit_α)
    ll = log_eppf(nj, α, θ; n=n, k=k)
    for i in 1:max_iter

        ∂α = (
            k / α + (digamma(θ / α + k) - digamma(θ / α)) * (-θ / α^2) -
            sum(digamma.(nj .- α) .- digamma(1 - α))
        )

        ∂θ = (
            (digamma(θ / α + k) - digamma(θ / α)) / α - digamma(θ + n) + digamma(θ)
        )

        logit_α += η * ∂α * α * (1 - α)
        α = logistic(logit_α)

        log_θ += η * ∂θ * θ
        θ = exp(log_θ)

        ll_new = log_eppf(nj, α, θ; n=n, k=k)

        if verbose && i % 1000 == 0
            println(f"step {i} ll {ll:.4e} α {α:.4f} θ {θ:.4f}")
        end

        if abs((ll_new - ll) / ll) < thres
            if verbose
                println(f"converged: step {i} ll {ll:.4e} α {α:.4f} θ {θ:.4f}")
            end
            break
        end

    end

    return α, θ
end

function pyp_rwmh(nj::Vector{Int}, num_iter::Int)

    α_rwmh = RandomWalkMetropolisHastings(SigmoidBijection())
    initialize!(α_rwmh, transform(randn(), α_rwmh.f))
    θ_rwmh = RandomWalkMetropolisHastings(ExpBijection())
    initialize!(θ_rwmh, transform(randn(), θ_rwmh.f))
    n, k = sum(nj), length(nj)



    for i in 1:num_iter

        log_f_α(x) = log_eppf(nj, x, get(θ_rwmh); n=n, k=k)
        mh_step!(α_rwmh, log_f_α)

        log_f_θ(x) = log_eppf_const(n, k, get(α_rwmh), x)
        mh_step!(θ_rwmh, log_f_θ)

    end

    return get(α_rwmh), get(θ_rwmh)

end