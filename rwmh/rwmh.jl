include("bijections.jl")
using Statistics
import Distributions: Normal, logpdf
import LinearAlgebra: I, cholesky, UniformScaling, LowerTriangular


mutable struct RandomWalkMetropolisHastings

    step::Int
    dim::Int
    scale::Float64
    α::Float64
    x::Union{Float64,Vector}
    μ::Union{Float64,Vector}
    Σ::Union{Float64,UniformScaling,Matrix}
    L::Union{Float64,LowerTriangular}
    nacc::Int
    f::AbstractBijection
    x_lb::Float64
    x_ub::Float64
    prior_mean::Float64
    prior_std::Float64

    function RandomWalkMetropolisHastings(
        f::AbstractBijection;
        dim::Int=1,
        scale::Float64=1.0,
        α::Float64=0.6,
        y_lb::Float64=-Inf,
        y_ub::Float64=Inf,
        prior_mean::Float64=0.0,
        prior_std::Float64=1.0
    )

        if y_lb == -Inf
            x_lb = -Inf
        else
            x_lb = inv_transform(y_lb, f)
        end

        if y_ub == Inf
            x_ub = Inf
        else
            x_ub = inv_transform(y_ub, f)
        end

        return new(0, dim, scale, α,
            0.0, 0.0, 1.0, 1.0, 0,
            f, x_lb, x_ub, prior_mean, prior_std)
    end

    function RandomWalkMetropolisHastings()
        return RandomWalkMetropolisHastings(IdentityBijection())
    end

end

function initialize!(rwmh::RandomWalkMetropolisHastings, y::Union{Float64,Vector})
    rwmh.step = 0
    rwmh.dim = length(y)
    rwmh.x = inv_transform(y, rwmh.f)
    rwmh.μ = copy(rwmh.x)
    rwmh.Σ = I
    rwmh.L = 1.0
    rwmh.nacc = 0
end

function update!(rwmh::RandomWalkMetropolisHastings, x::Union{Float64,Vector})
    rwmh.step += 1
    γ = (rwmh.step + 1)^(-rwmh.α)
    rwmh.x = copy(x)
    rwmh.μ = (1 - γ) * rwmh.μ + γ * rwmh.x
    diff = rwmh.x - rwmh.μ
    rwmh.Σ = (1 - γ) * rwmh.Σ + γ * (diff * diff')
    rwmh.L = cholesky(rwmh.Σ).L
end

function get(rwmh::RandomWalkMetropolisHastings)
    return transform(rwmh.x, rwmh.f)
end

function acc_rate(rwmh::RandomWalkMetropolisHastings)
    return rwmh.nacc / rwmh.step
end

function propose(rwmh::RandomWalkMetropolisHastings)

    while true
        L = rwmh.scale * 2.38 / sqrt(rwmh.dim) * rwmh.L
        ε = L * randn(rwmh.dim)
        if isa(rwmh.x, Float64)
            ε = only(ε)
        end
        x_new = rwmh.x + ε

        if (x_new > rwmh.x_lb) & (x_new < rwmh.x_ub)
            return x_new
        end

    end

end

# function transformed_log_prior(y::Union{Float64, Vector}, f::AbstractBijection)
#     return sum(logpdf.(inv_transform(y, f))) + log_jacobian(y, f)
# end

# log_likel is a function taking y to evaluate the likelihood
# if log_prior is not given, it is set to be (transformed) unit Gaussian.
function mh_step!(rwmh::RandomWalkMetropolisHastings, log_likel::Any; log_prior::Any=nothing)

    x = rwmh.x
    y = transform(x, rwmh.f)
    x_new = propose(rwmh)
    y_new = transform(x_new, rwmh.f)

    if log_prior === nothing
        px = Normal(rwmh.prior_mean, rwmh.prior_std)
        log_ρ = (
            log_likel(y_new) + sum(logpdf.(px, x_new)) -
            log_likel(y) - sum(logpdf.(px, x))
        )
    else
        log_ρ = (
            log_likel(y_new) + log_prior(y_new) + log_jacobian(y, rwmh.f) -
            log_likel(y) - log_prior(y) - log_jacobian(y_new, rwmh.f)
        )
    end

    if rand() < exp(min(log_ρ, 0.0))
        update!(rwmh, x_new)
        rwmh.nacc += 1
    else
        update!(rwmh, x)
    end

end
