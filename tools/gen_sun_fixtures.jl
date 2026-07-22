#!/usr/bin/env julia
# Generate exact SU(N) combinatorics fixtures for the racah crate's
# cgc-gen Layer 1 oracle (issue #10), from SUNRepresentations.jl v0.4.0.
#
# One tagged line per record (';'-separated fields):
#   DIM;N;dynkin;dim;dualdynkin
#         Weyl dimension and dual, dynkin/dualdynkin comma-separated (N-1 ints).
#   PAT;N;dynkin;d1,d2,...
#         One GT pattern (flat data, top row first) per line, emitted in the
#         reference basis() order. Consecutive PAT lines with the same
#         (N,dynkin) give that irrep's ordered pattern list.
#   DP;N;a_dynkin;b_dynkin;c_dynkin:mult|c_dynkin:mult|...
#         Littlewood-Richardson decomposition of a (x) b.
#   LAD;N;dynkin;l,i,j,num,den
#         One nonzero creation-matrix entry: level l (1..N-1), 1-based basis
#         indices (i,j), and signedsquare(value) = num/den (num signed, den>0).
#
# Provenance is emitted as leading '#' comment lines. Values are produced by
# SUNRepresentations, never fabricated.
#
# Usage (needs an environment with SUNRepresentations v0.4.0 + RationalRoots):
#   julia --project=<env> tools/gen_sun_fixtures.jl > tests/fixtures/sun/sun_fixtures.txt

using SUNRepresentations
using RationalRoots
using Dates
using SHA
using Random

# dual via reversed Dynkin labels (sector.jl:dual), avoiding a TensorKitSectors
# import in this script.
dualweight(s) = weight(typeof(s)(reverse(dynkin_label(s))))
dynkin_csv(v) = join(v, ",")

# All Dynkin labels for SU(N) with each component in 0:maxc and dim <= dimcap.
function sample_irreps(N, maxc, dimcap)
    out = SUNIrrep{N}[]
    for tup in Iterators.product(ntuple(_ -> 0:maxc, N - 1)...)
        s = SUNIrrep{N}(tup...)  # N-1 args -> Dynkin constructor
        dim(s) <= dimcap && push!(out, s)
    end
    return out
end

# Per-rank sampling budgets: keep N=5 small (dims grow fast), and cap the
# pattern/ladder dimension so the fixture stays compact but still exercises
# multi-row GT recursion.
const RANK_CONF = Dict(
    2 => (maxc = 9, dimcap = 10_000, patcap = 8, ladcap = 8, dppairs = 60, dpcap = 10),
    3 => (maxc = 3, dimcap = 10_000, patcap = 64, ladcap = 27, dppairs = 60, dpcap = 200),
    4 => (maxc = 2, dimcap = 10_000, patcap = 64, ladcap = 20, dppairs = 60, dpcap = 200),
    5 => (maxc = 1, dimcap = 10_000, patcap = 64, ladcap = 30, dppairs = 60, dpcap = 200),
)

function emit_dim_dual(io, s)
    N = rank(s)
    println(io, "DIM;", N, ";", dynkin_csv(dynkin_label(s)), ";", dim(s), ";",
        dynkin_csv(_weight_to_dynkin_ints(dualweight(s))))
end

# dual returns a weight tuple; convert to Dynkin (differences) for the label.
_weight_to_dynkin_ints(w) = [w[i] - w[i + 1] for i in 1:(length(w) - 1)]

function emit_patterns(io, s)
    N = rank(s)
    dk = dynkin_csv(dynkin_label(s))
    for m in basis(s)
        println(io, "PAT;", N, ";", dk, ";", join(m.data, ","))
    end
end

function emit_directproduct(io, s1, s2)
    N = rank(s1)
    dp = directproduct(s1, s2)
    parts = String[]
    for (c, mult) in dp
        push!(parts, string(dynkin_csv(dynkin_label(c)), ":", mult))
    end
    println(io, "DP;", N, ";", dynkin_csv(dynkin_label(s1)), ";",
        dynkin_csv(dynkin_label(s2)), ";", join(parts, "|"))
end

function emit_ladder(io, s)
    N = rank(s)
    dk = dynkin_csv(dynkin_label(s))
    c = creation(s)
    for l in 1:(N - 1)
        A = Array(c[l])
        d = size(A, 1)
        for j in 1:d, i in 1:d
            v = A[i, j]
            iszero(v) && continue
            sq = signedsquare(v)              # Rational: sign(coef)*coef... == coef
            println(io, "LAD;", N, ";", dk, ";", l, ",", i, ",", j, ",",
                numerator(sq), ",", denominator(sq))
        end
    end
end

function main()
    script = @__FILE__
    hash = bytes2hex(sha256(read(script)))
    println("# racah SU(N) combinatorics fixtures (cgc-gen Layer 1, issue #10)")
    println("# generator: tools/gen_sun_fixtures.jl")
    println("# SUNRepresentations.jl version: ", pkgversion(SUNRepresentations))
    println("# RationalRoots.jl version: ", pkgversion(RationalRoots))
    println("# julia version: ", VERSION)
    println("# generated (UTC): ", Dates.now(Dates.UTC))
    println("# script sha256: ", hash)
    println("# record tags: DIM PAT DP LAD  (see script header for field layout)")

    rng = MersenneTwister(0x5D5E9C7A3B1F2604)
    dp_total = 0
    for N in 2:5
        conf = RANK_CONF[N]
        irreps = sample_irreps(N, conf.maxc, conf.dimcap)

        # DIM + dual for every sampled irrep.
        for s in irreps
            emit_dim_dual(io_stdout(), s)
        end

        # Pattern-order pin for the small-dim irreps.
        for s in irreps
            dim(s) <= conf.patcap && emit_patterns(io_stdout(), s)
        end

        # Ladder entries for the small-dim irreps.
        for s in irreps
            dim(s) <= conf.ladcap && emit_ladder(io_stdout(), s)
        end

        # directproduct: random pairs from the small-dim pool (cheap basis
        # iteration). Cap the target at the number of distinct ordered pairs.
        pool = filter(s -> dim(s) <= conf.dpcap, irreps)
        n = length(pool)
        target = min(conf.dppairs, n * n)
        seen = Set{Tuple{Int,Int}}()
        while length(seen) < target
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
            (i, j) in seen && continue
            push!(seen, (i, j))
            emit_directproduct(io_stdout(), pool[i], pool[j])
            dp_total += 1
        end
    end
    @assert dp_total >= 200 "expected >= 200 directproduct cases, got $dp_total"
    return
end

io_stdout() = stdout

main()
