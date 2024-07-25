
# Types inherit from AbstractTransform should provide
# function transform(x): computes y = f(x)
# function inv_transform(x): computes x = f^-1(y)
# function log_jacobian(y): computes |∂x/∂y| so that log_pdf_y(y) = log_pdf_x(f^-1(y)) + log |∂x/∂y|
abstract type AbstractBijection end

struct IdentityBijection <: AbstractBijection
end

function transform(x::Float64, f::IdentityBijection)
    return x
end

function transform(x::Vector, f::IdentityBijection)
    return x
end

function inv_transform(y::Float64, f::IdentityBijection)
    return y
end

function inv_transform(y::Vector, f::IdentityBijection)
    return y
end

function log_jacobian(y::Float64, f::IdentityBijection)
    return 0.0
end

function log_jacobian(y::Vector, f::IdentityBijection)
    return 0.0
end

# y = x + t
struct TranslateBijection <: AbstractBijection
    t::Float64
end

function transform(x::Float64, f::TranslateBijection)
    return x + f.t
end

function transform(x::Vector, f::TranslateBijection)
    return x .+ f.t
end

function inv_transform(y::Float64, f::TranslateBijection)
    return y - f.t
end

function inv_transform(y::Vector, f::TranslateBijection)
    return y .- f.t
end

function log_jacobian(y::Float64, f::TranslateBijection)
    return 0.0
end

function log_jacobian(y::Vector, f::TranslateBijection)
    return 0.0
end

# y = exp(x)
struct ExpBijection <: AbstractBijection

end

function transform(x::Float64, f::ExpBijection)
    return exp(x)
end

function transform(x::Vector, f::ExpBijection)
    return exp.(x)
end

function inv_transform(y::Float64, f::ExpBijection)
    return log(y)
end

function inv_transform(y::Vector, f::ExpBijection)
    return log.(y)
end

function log_jacobian(y::Float64, f::ExpBijection)
    return -log(y)
end

function log_jacobian(y::Vector, f::ExpBijection)
    return -sum(log.(y))
end

# y = scale/(1 + exp(-x))
struct SigmoidBijection <: AbstractBijection
    scale::Float64
    function SigmoidBijection()
        return new(1.0)
    end
    function SigmoidBijection(scale::Float64)
        return new(scale)
    end
end

function transform(x::Float64, f::SigmoidBijection)
    return f.scale / (1 + exp(-x))
end

function transform(x::Vector, f::SigmoidBijection)
    return f.scale ./ (1 .+ exp.(-x))
end

function inv_transform(y::Float64, f::SigmoidBijection)
    return log(y) - log(f.scale - y)
end

function inv_transform(y::Vector, f::SigmoidBijection)
    return log.(y) - log.(f.scale .- y)
end

function log_jacobian(y::Float64, f::SigmoidBijection)
    return log(f.scale) - log(y) - log(f.scale - y)
end

function log_jacobian(y::Vector, f::SigmoidBijection)
    return sum(log.(f.scale) .- log.(y) .- log.(f.scale .- y))
end

# y = f_d ∘ … ∘ f_1 (x) 
struct CompositeBijection <: AbstractBijection
    bijections::Vector{AbstractBijection}
    function CompositeBijection(bijections::Vararg{AbstractBijection})
        return new(collect(bijections))
    end
end

function transform(x::Float64, f::CompositeBijection)
    y = transform(x, f.bijections[1])
    for fj in f.bijections[2:end]
        y = transform(y, fj)
    end
    return y
end

function transform(x::Vector, f::CompositeBijection)
    y = transform(x, f.bijections[1])
    for fj in f.bijections[2:end]
        y = transform(y, fj)
    end
    return y
end

function inv_transform(y::Float64, f::CompositeBijection)
    x = inv_transform(y, f.bijections[end])
    for fj in reverse(f.bijections[1:end-1])
        x = inv_transform(x, fj)
    end
    return x
end

function inv_transform(y::Vector, f::CompositeBijection)
    x = inv_transform(y, f.bijections[end])
    for fj in reverse(f.bijections[1:end-1])
        x = inv_transform(x, fj)
    end
    return x
end

function log_jacobian(y::Float64, f::CompositeBijection)
    log_jac = log_jacobian(y, f.bijections[end])
    x = inv_transform(y, f.bijections[end])
    for fj in reverse(f.bijections[1:end-1])
        log_jac += log_jacobian(x, fj)
        x = inv_transform(x, fj)
    end
    return log_jac
end

function log_jacobian(y::Vector, f::CompositeBijection)
    log_jac = log_jacobian(y, f.bijections[end])
    x = inv_transform(y, f.bijections[end])
    for fj in reverse(f.bijections[1:end-1])
        log_jac += log_jacobian(x, fj)
        x = inv_transform(x, fj)
    end
    return log_jac
end

# [y_1, … y_d] = [f_1(x_1), …, f_d(x_d)]
struct ConcatBijection <: AbstractBijection
    bijections::Vector{AbstractBijection}
    function ConcatBijection(bijections::Vararg{AbstractBijection})
        return new(collect(bijections))
    end
end

function transform(x::Vector, f::ConcatBijection)
    y = zero(x)
    for (i, fj) in enumerate(f.bijections)
        y[i] = transform(x[i], fj)
    end
    return y
end

function inv_transform(y::Vector, f::ConcatBijection)
    x = zero(y)
    for (i, fj) in enumerate(f.bijections)
        x[i] = inv_transform(y[i], fj)
    end
    return x
end

function log_jacobian(y::Vector, f::ConcatBijection)
    log_jac = 0.0
    for (i, fj) in enumerate(f.bijections)
        log_jac += log_jacobian(y[i], fj)
    end
    return log_jac
end
