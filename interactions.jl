# using DataStructures: Accumulator, counter

mutable struct Node
    index::Int
    parent::Union{Nothing,Node}
    children::Vector{Node}
    function Node(index::Int)
        return new(index, nothing, [])
    end
    function Node(index::Int, parent::Node)
        return new(index, parent, [])
    end
end

mutable struct Interactions
    nodes::Vector{Node}
    edge_sequence::Vector{Int} # a single vector formed by concatenating all the edges
    edge_lengths::Vector{Int}
    edge_length_freqs::Dict{Int,Int}
    degrees::Vector{Int}

    function Interactions(nodes::Vector{Node},
        edge_sequence::Vector{Int},
        edge_lengths::Vector{Int};
        edge_length_freqs::Union{Nothing,Dict{Int,Int}}=nothing,
        degrees::Union{Nothing,Vector{Int}}=nothing)

        if edge_length_freqs === nothing
            edge_length_freqs = Dict{Int,Int}()
            for c in edge_lengths
                if haskey(edge_length_freqs, c)
                    edge_length_freqs[c] += 1
                else
                    edge_length_freqs[c] = 1
                end
            end
        end

        if degrees === nothing
            degrees = zeros(Int, length(nodes))
            for j in edge_sequence
                degrees[j] += 1
            end
        end

        return new(nodes, edge_sequence, edge_lengths, edge_length_freqs, degrees)
    end

end

function get_partition(In::Interactions)
    partition = Vector{Vector{Int}}(undef, length(In.nodes))
    for j in eachindex(partition)
        partition[j] = zeros(Int, 0)
    end
    for (i, j) in enumerate(In.edge_sequence)
        push!(partition[j], i)
    end
    return partition
end