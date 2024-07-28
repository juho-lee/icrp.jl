using Serialization
using ArgParse
include("../pdgm_coag_icrp.jl")
include("../utils.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--num_chains"
        help = "number of chains to run"
        arg_type = Int
        default = 3
        "--num_steps"
        help = "number of steps per chain"
        arg_type = Int
        default = 50000
        "--print_every"
        help = "print sampler status at every"
        arg_type = Int
        default = 10000
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Parsed args:")
    for (arg, val) in args
        println("  $arg => $val")
    end

    In, coag_In = deserialize("../wikivote/wikivote_coag.data")

    path = "./results/pdgm_coag_icrp"
    mkpath(path)

    for i in 1:args["num_chains"]
        println("Running chain $i...")
        chain = run_sampler(In, coag_In, args["num_steps"]; print_every=args["print_every"])
        pred = simulate_predictives(In, coag_In, chain)
        serialize(joinpath(path, "run$i.data"), [chain, pred])
    end
end

main()