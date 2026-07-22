#!/usr/bin/env julia
# Generate exact Wigner 6j fixtures beyond the `wigner-symbols 0.5.1` domain
# (doubled spins > 254), for the racah crate's large-fixture test.
#
# Each line is:  dj1 dj2 dj3 dj4 dj5 dj6 sign num den
# where the exact value satisfies  value^2 = num/den  and  sign in {-1,0,1};
# i.e. signedsquare(value) = sign * num/den (num/den reduced, nonnegative).
#
# Usage:
#   julia tools/gen_fixtures.jl > tests/fixtures/su2_6j_large.txt
#
# Requires WignerSymbols.jl and RationalRoots.jl from the local registry.

using WignerSymbols
using RationalRoots
using Dates
using SHA
using Random

triangle_ok(a, b, c) = iseven(a + b + c) && c >= abs(a - b) && c <= a + b

function admissible6j(dj)
    dj1, dj2, dj3, dj4, dj5, dj6 = dj
    triangle_ok(dj1, dj2, dj3) &&
        triangle_ok(dj1, dj5, dj6) &&
        triangle_ok(dj4, dj2, dj6) &&
        triangle_ok(dj4, dj5, dj3)
end

# Deterministically search for admissible doubled-spin 6j label sets with all
# twice-spins in `range`. Seeded for reproducibility (the provenance header
# records the Julia version that fixes the RNG stream).
function search_cases(rng, range, count)
    out = Vector{NTuple{6,Int}}()
    seen = Set{NTuple{6,Int}}()
    while length(out) < count
        dj = ntuple(_ -> rand(rng, range), 6)
        if admissible6j(dj) && !(dj in seen)
            push!(seen, dj)
            push!(out, dj)
        end
    end
    out
end

# Fixed list of doubled-spin 6j label sets, all beyond the reference-crate
# domain: ~30 with twice-spins in 255..600, plus a handful in the thousands.
# The thousands band is kept to 1000..1400: still comfortably past the u8
# reference ceiling, while keeping the (ignored, on-demand) huge-tier test's
# runtime to minutes rather than tens of minutes, since the Racah sum's
# factorial sizes grow steeply with the spin.
const CASES = let
    rng = MersenneTwister(0x9E3779B97F4A7C15)
    v = search_cases(rng, 255:600, 30)
    append!(v, search_cases(rng, 1000:1400, 6))
    v
end

function main()
    script = @__FILE__
    hash = bytes2hex(sha256(read(script)))
    println("# racah SU(2) 6j large fixtures")
    println("# generator: tools/gen_fixtures.jl")
    println("# WignerSymbols.jl version: ", pkgversion(WignerSymbols))
    println("# RationalRoots.jl version: ", pkgversion(RationalRoots))
    println("# julia version: ", VERSION)
    println("# generated (UTC): ", Dates.now(Dates.UTC))
    println("# script sha256: ", hash)
    println("# columns: dj1 dj2 dj3 dj4 dj5 dj6 sign num den   (value^2 = num/den)")
    for dj in CASES
        @assert admissible6j(dj) "non-admissible fixture case $dj"
        # WignerSymbols uses half-integer spins; convert doubled -> Rational.
        js = map(x -> x // 2, dj)
        val = wigner6j(js...)            # RationalRoot
        ss = signedsquare(val)           # Rational: sign * value^2
        s = sign(ss)
        r = abs(ss)
        println(join(dj, " "), " ", Int(s), " ", numerator(r), " ", denominator(r))
    end
end

main()
