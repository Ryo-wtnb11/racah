# Independent-implementation oracle fixtures for racah's B/C/D combinatorics,
# OSCAR variant (alternative to tools/gen_bcd_fixtures.py). Run with
#   julia --project tools/gen_bcd_fixtures.jl
# in an environment where Oscar.jl is installed.
#
# Uses Oscar's Lie-theory layer: a simple Lie algebra of the requested Cartan
# type, highest-weight modules addressed by integer Dynkin labels (the same
# convention as racah::bcd), exact dimensions, and the tensor-product
# multiplicities. Output format matches the Sage script exactly:
#
#   SERIES RANK | dynkin_a | dynkin_b | dim_a dim_b | c1:n1 c2:n2 ...
#
# with a provenance header (Oscar version + script sha256). Only tensor irreps
# are generated; spinor labels are never emitted.

using Oscar
using SHA
using Random

const CASES = [("B", 2), ("B", 3), ("B", 4),
               ("C", 2), ("C", 3), ("C", 4),
               ("D", 3), ("D", 4)]
const PAIRS_PER_CASE = 8
const MAX_LABEL = Dict(2 => 3, 3 => 2, 4 => 1)
const SEED = 0x0BCD0019

# Force a Dynkin label onto the tensor (integer-weight) sublattice.
function tensorize(series, a)
    a = collect(a); r = length(a)
    if series == "B"
        a[r] -= a[r] % 2
    elseif series == "D"
        if (a[r-1] + a[r]) % 2 != 0
            a[r] += a[r] < MAX_LABEL[r] ? 1 : -1
        end
    end
    return a
end

rand_label(rng, series, r) = tensorize(series, [rand(rng, 0:MAX_LABEL[r]) for _ in 1:r])

cartan_symbol(s) = s == "B" ? :B : s == "C" ? :C : :D

function main()
    rng = MersenneTwister(SEED)
    sha = bytes2hex(sha256(read(@__FILE__)))
    out = String[]
    push!(out, "--- racah B/C/D fixtures (independent oracle)")
    push!(out, "--- tool: Oscar $(pkgversion(Oscar))")
    push!(out, "--- script: gen_bcd_fixtures.jl sha256=$sha")
    push!(out, "--- seed: $(SEED)  format: SERIES RANK | a | b | dim_a dim_b | c:n ...")

    for (series, r) in CASES
        L = lie_algebra(QQ, cartan_symbol(series), r)
        for _ in 1:PAIRS_PER_CASE
            da = rand_label(rng, series, r)
            db = rand_label(rng, series, r)
            Va = simple_module(L, da)
            Vb = simple_module(L, db)
            # Decompose the tensor product into simple highest-weight modules.
            dec = tensor_product_decomposition(L, da, db)
            terms = String[]
            for (wt, mult) in dec
                push!(terms, join(Int.(wt), ",") * ":" * string(Int(mult)))
            end
            sort!(terms)
            push!(out, "$series $r | $(join(da, ",")) | $(join(db, ",")) | " *
                       "$(dim(Va)) $(dim(Vb)) | $(join(terms, " "))")
        end
    end

    dest = joinpath(@__DIR__, "..", "tests", "fixtures")
    mkpath(dest)
    path = joinpath(dest, "bcd_fixtures.json")
    open(path, "w") do io
        write(io, join(out, "\n") * "\n")
    end
    @info "wrote $path ($(length(out) - 4) records)"
end

main()
