
using Measures: mm
import Statistics: mean, std
using PyFormattedStrings
using LaTeXStrings
using MCMCDiagnosticTools

function plot_chain(chain::ICRPChain;
    true_vals::Union{Dict{String,Float64},Nothing}=nothing,
    latex_codes::Union{Dict{String,String},Nothing}=nothing,
    rhats::Union{Dict{String,Float64},Nothing}=nothing)

    ps = []

    if true_vals === nothing
        true_vals = Dict{String,Float64}()
    end

    if latex_codes === nothing
        latex_codes = Dict{String, String}()
    end

    if rhats === nothing
        rhats = Dict{String,Float64}()
    end

    rhat = Base.get(rhats, "α", nothing)
    code = Base.get(latex_codes, "α", "\\alpha")
    if rhat !== nothing
        title = L"%$(code)\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"%$(code)"
    end
    p = histogram(chain.α,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title,
        xformatter=x -> string(f"{x:.2f}"))
    if Base.get(true_vals, "α", nothing) !== nothing
        vline!([true_vals["α"]], label=nothing, color="black",
            linewidth=1.5, linestyle=:dash)
    end
    push!(ps, p)

    rhat = Base.get(rhats, "θ", nothing)
    code = Base.get(latex_codes, "θ", "\\theta")
    if rhat !== nothing
        title = L"%$(code)\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"%$(code)"
    end
    p = histogram(chain.θ,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title)
    if Base.get(true_vals, "θ", nothing) !== nothing
        vline!([true_vals["θ"]], label=nothing, color="black",
            linewidth=1.5, linestyle=:dash)
    end
    push!(ps, p)

    rhat = Base.get(rhats, "t", nothing)
    code = Base.get(latex_codes, "t", "t")
    if rhat !== nothing
        title = L"%$(code)\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"%$(code)"
    end
    p = histogram(chain.t,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title)
    push!(ps, p)

    p = plot(ps..., layout=(1, 3), size=(1000, 300), margin=5mm)
    return p
end

function plot_chain(chains::Vector{ICRPChain};
    latex_codes::Union{Dict{String,String},Nothing}=nothing,
    true_vals::Union{Dict{String,Float64},Nothing}=nothing)
    merged_chain = ICRPChain()
    for chain in chains
        append!(merged_chain.α, chain.α)
        append!(merged_chain.θ, chain.θ)
        append!(merged_chain.t, chain.t)
        append!(merged_chain.log_ν, chain.log_ν)
    end
    rhats = Dict{String,Float64}()
    rhats["α"] = rhat(hcat([chain.α for chain in chains]...))
    rhats["θ"] = rhat(hcat([chain.θ for chain in chains]...))
    rhats["t"] = rhat(hcat([chain.t for chain in chains]...))

    plot_chain(merged_chain; true_vals=true_vals, latex_codes=latex_codes, rhats=rhats)
end

function plot_predictions(In::Interactions, pred::ICRPPred)

    num_nodes = length(In.nodes)
    num_edges = length(In.edge_lengths)
    # node_rrmse = sqrt(mean(((pred.num_nodes .- num_nodes) ./ num_nodes) .^ 2))
    # edge_rrmse = sqrt(mean(((pred.num_edges .- num_edges) ./ num_edges) .^ 2))

    node_rmse = sqrt(mean((pred.num_nodes .- num_nodes) .^ 2))
    edge_rmse = sqrt(mean((pred.num_edges .- num_edges) .^ 2))


    line = (
        f"Number of nodes RMSE: {node_rmse:.4f}\n" *
        f"Number of edges RMSE: {edge_rmse:.4f}\n" *
        f"Degree dist average KS: {mean(pred.degree_kss):.4f}+-{std(pred.degree_kss):.4f}"
    )
    println(line)

    ps = []

    p = plot_fof(In.degrees)
    plot_fof_CI(pred.degrees)
    push!(ps, p)

    p = histogram(pred.num_nodes,
        label=nothing,
        color="skyblue",
        linecolor="steelblue",
        title="Number of nodes",
        titlefont=font(15, "Helvetica"),
        xrotation=45
    )
    vline!([length(In.nodes)], label=nothing, color="black", linewidth=1.5, linestyle=:dash)
    push!(ps, p)

    p = histogram(pred.num_edges,
        label=nothing,
        color="skyblue",
        linecolor="steelblue",
        title="Number of edges",
        titlefont=font(15, "Helvetica"),
        xrotation=45)
    vline!([length(In.edge_lengths)], label=nothing, color="black", linewidth=1.5, linestyle=:dash)
    push!(ps, p)

    p = plot(ps..., layout=(1, 3), size=(900, 300),
        bottom_margin=7mm, top_margin=5mm, left_margin=2mm, right_margin=2mm)
    return p

end

function plot_predictions(In::Interactions, preds::Vector{ICRPPred})
    merged_pred = ICRPPred()
    for pred in preds
        append!(merged_pred.degrees, pred.degrees)
        append!(merged_pred.num_edges, pred.num_edges)
        append!(merged_pred.num_nodes, pred.num_nodes)
        append!(merged_pred.degree_kss, pred.degree_kss)
    end
    plot_predictions(In, merged_pred)
end