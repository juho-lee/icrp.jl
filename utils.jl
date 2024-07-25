using Plots
using StatsBase
using Statistics
using LaTeXStrings


function plot_fof(freq;
    xloglim::Int=16,
    colorset::String="blue",
    redraw::Bool=false,
    asymp_func::Union{Function,Nothing}=nothing,
    label::Union{String,LaTeXString,Nothing}=nothing
)

    edgebins = [2^i for i in 0:xloglim]
    sizebins = edgebins[2:end] - edgebins[1:end-1]
    append!(sizebins, 1)
    centerbins = edgebins
    hist = fit(Histogram, freq, edgebins)
    counts = vcat(hist.weights, [0])
    fof = counts ./ sizebins ./ length(freq)
    indices = fof .> 0
    centerbins = centerbins[indices]
    fof = fof[indices]

    if redraw
        plot_fn = plot!
    else
        plot_fn = plot
    end

    if colorset == "blue"
        markershape = :circle
        markersize = 5
        markercolor = "skyblue"
        markerstrokecolor = "blue"
        linecolor = "steelblue"
        linewidth = 1.5
    elseif colorset == "red"
        markershape = :diamond
        markersize = 5
        markercolor = "tomato"
        markerstrokecolor = "darkred"
        linecolor = "indianred"
        linewidth = 1.5
    elseif colorset == "green"
        markershape = :utriangle
        markersize = 5
        markercolor = "yellowgreen"
        markerstrokecolor = "seagreen"
        linecolor = "limegreen"
        linewidth = 1.5
    end

    p = plot_fn(centerbins, fof, xaxis=:log, yaxis=:log, label=label,
        markershape=markershape,
        markersize=markersize,
        markercolor=markercolor,
        markerstrokecolor=markerstrokecolor,
        linecolor=linecolor,
        linewidth=linewidth,
        # legendfontsize=12
        legendfont=font(12, "Helvetica")
    )

    if asymp_func !== nothing
        asymp_fofs = zeros(length(centerbins))
        for (i, cb) in enumerate(centerbins)
            asymp_fofs[i] = asymp_func(cb)
        end
        plot!(centerbins, asymp_fofs, label=nothing,
            linestyle=:dash,
            linecolor=linecolor,
            linewidth=linewidth
        )
    end


    return p
end

function plot_fof_CI(freqs;
    redraw::Bool=true,
    xloglim::Int=16,
    α::Float64=0.95,
    colorset::String="blue"
)

    edgebins = [2^i for i in 0:xloglim]
    sizebins = edgebins[2:end] - edgebins[1:end-1]
    append!(sizebins, 1)
    centerbins = edgebins
    fofs = zeros(length(edgebins), length(freqs))
    for (i, freq) in enumerate(freqs)
        hist = fit(Histogram, freq, edgebins)
        counts = vcat(hist.weights, [0])
        fof = counts ./ sizebins ./ length(freq)
        fofs[:, i] = fof
    end

    lbs = zeros(length(edgebins))
    ubs = zeros(length(edgebins))
    for i in 1:length(edgebins)
        lbs[i], ubs[i] = quantile(fofs[i, :], [0.5 * (1 - α), 0.5 * (1 + α)])
    end

    indices = (ubs .> 0) .& (lbs .> 0)
    lbs = lbs[indices]
    ubs = ubs[indices]
    centerbins = centerbins[indices]

    if colorset == "blue"
        fillcolor = "skyblue"
    elseif colorset == "red"
        fillcolor = "tomato"
    end


    if redraw
        plot_fn = plot!
    else
        plot_fn = plot
    end

    p = plot_fn(centerbins,
        lbs,
        fillrange=ubs,
        label=nothing,
        linecolor=nothing,
        fillcolor=fillcolor,
        fillalpha=0.3,
        xaxis=:log,
        yaxis=:log)

    return p
end
