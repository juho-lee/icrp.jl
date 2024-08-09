using Measures: mm
import Statistics: mean, std
using PyFormattedStrings
using LaTeXStrings
using MCMCDiagnosticTools

function plot_chain(chain::PDGMCoagICRPChain;
    true_vals::Union{Dict{String,Float64},Nothing}=nothing,
    rhats::Union{Dict{String,Float64},Nothing}=nothing)

    ps = []

    if true_vals === nothing
        true_vals = Dict{String,Float64}()
    end

    if rhats === nothing
        rhats = Dict{String,Float64}()
    end

    rhat = Base.get(rhats, "α", nothing)
    if rhat !== nothing
        title = L"\alpha\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"\alpha"
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


    rhat = Base.get(rhats, "β", nothing)
    if rhat !== nothing
        title = L"\beta\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"\beta"
    end
    p = histogram(chain.β,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title)

    if Base.get(true_vals, "β", nothing) !== nothing
        vline!([true_vals["β"]], label=nothing, color="black",
            linewidth=1.5, linestyle=:dash)
    end
    push!(ps, p)

    rhat = Base.get(rhats, "θ", nothing)
    if rhat !== nothing
        title = L"\theta\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"\theta"
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

    rhat = Base.get(rhats, "m", nothing)
    if rhat !== nothing
        title = L"m\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"m"
    end
    p = histogram(chain.m,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title)
    if Base.get(true_vals, "m", nothing) !== nothing
        vline!([true_vals["m"]], label=nothing, color="black",
            linewidth=1.5, linestyle=:dash)
    end
    push!(ps, p)

    rhat = Base.get(rhats, "t", nothing)
    if rhat !== nothing
        title = L"\gamma\quad(\hat{R}=%$(round(rhat;digits=4)))"
    else
        title = L"\gamma"
    end
    p = histogram(chain.t,
        label=nothing,
        color="lightsalmon",
        linecolor="tomato2",
        title=title)
    push!(ps, p)

    p = plot(ps..., layout=(1, 5), size=(1600, 250), margin=3mm)
    return p
end


function plot_chain(chains::Vector{PDGMCoagICRPChain};
    true_vals::Union{Dict{String,Float64},Nothing}=nothing)
    merged_chain = PDGMCoagICRPChain()
    for chain in chains
        append!(merged_chain.α, chain.α)
        append!(merged_chain.β, chain.β)
        append!(merged_chain.θ, chain.θ)
        append!(merged_chain.m, chain.m)
        append!(merged_chain.t, chain.t)
        append!(merged_chain.log_ν, chain.log_ν)
    end

    rhats = Dict{String,Float64}()
    rhats["α"] = rhat(hcat([chain.α for chain in chains]...))
    rhats["β"] = rhat(hcat([chain.β for chain in chains]...))
    rhats["θ"] = rhat(hcat([chain.θ for chain in chains]...))
    rhats["m"] = rhat(hcat([chain.m for chain in chains]...))
    rhats["t"] = rhat(hcat([chain.t for chain in chains]...))

    plot_chain(merged_chain; true_vals=true_vals, rhats=rhats)
end

function plot_predictions(In::Interactions, coag_In::Interactions, pred::PDGMCoagICRPPred)

    num_nodes = length(In.nodes)
    coag_num_nodes = length(coag_In.nodes)
    num_edges = length(In.edge_lengths)

    node_rmse = sqrt(mean((pred.num_nodes .- num_nodes) .^ 2))
    coag_node_rmse = sqrt(mean((pred.coag_num_nodes .- coag_num_nodes) .^ 2))
    edge_rmse = sqrt(mean((pred.num_edges .- num_edges) .^ 2))

    line = (
        f"Number of nodes RMSE: {node_rmse:.4f}\n" *
        f"Number of edges RMSE: {edge_rmse:.4f}\n" *
        f"Degree dist average KS: {mean(pred.degree_kss):.4f}+-{std(pred.degree_kss):.4f}\n" *
        f"Coag number of nodes RMSE: {coag_node_rmse:.4f}\n" *
        f"Coag degree dist average KS: {mean(pred.coag_degree_kss):.4f}+-{std(pred.coag_degree_kss):.4f}"
    )
    println(line)

    ps = []

    p = plot_fof(In.degrees; label="Original")
    plot_fof_CI(pred.degrees)
    plot_fof(coag_In.degrees; redraw=true, colorset="red", label="Coag")
    plot_fof_CI(pred.coag_degrees; colorset="red")
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

    p = histogram(pred.coag_num_nodes,
        label=nothing,
        color="skyblue",
        linecolor="steelblue",
        title="Number of coag nodes",
        titlefont=font(15, "Helvetica"),
        xrotation=45
    )
    vline!([length(coag_In.nodes)], label=nothing, color="black", linewidth=1.5, linestyle=:dash)
    push!(ps, p)

    p = plot(ps..., layout=(1, 4), size=(1300, 280),
        bottom_margin=7mm, top_margin=5mm, left_margin=2mm, right_margin=2mm)
    return p

end

function plot_predictions(In::Interactions, coag_In::Interactions, preds::Vector{PDGMCoagICRPPred})
    merged_pred = PDGMCoagICRPPred()
    # total_num_samples = 0
    for pred in preds
        append!(merged_pred.degrees, pred.degrees)
        append!(merged_pred.num_edges, pred.num_edges)
        append!(merged_pred.num_nodes, pred.num_nodes)
        append!(merged_pred.degree_kss, pred.degree_kss)
        append!(merged_pred.coag_degrees, pred.coag_degrees)
        append!(merged_pred.coag_degree_kss, pred.coag_degree_kss)
        append!(merged_pred.coag_num_nodes, pred.coag_num_nodes)

        # num_samples = length(pred.degrees)
        # total_num_samples += num_samples
        # merged_pred.degrees_mks += num_samples * pred.degrees_mks
        # merged_pred.num_edges_rmse += num_samples * pred.num_edges_rmse
        # merged_pred.num_nodes_rmse += num_samples * pred.num_nodes_rmse
        # merged_pred.coag_degrees_mks += num_samples * pred.coag_degrees_mks
        # merged_pred.coag_num_nodes_rmse += num_samples * pred.coag_num_nodes_rmse
    end
    # merged_pred.degrees_mks /= total_num_samples
    # merged_pred.num_edges_rmse /= total_num_samples
    # merged_pred.num_nodes_rmse /= total_num_samples
    # merged_pred.coag_degrees_mks /= total_num_samples
    # merged_pred.coag_num_nodes_rmse /= total_num_samples
    plot_predictions(In, coag_In, merged_pred)
end