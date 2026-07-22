# Emit SU(3) F- and R-symbol fixtures for the racah table-regeneration oracle
# (issue #16, oracle 1). Values come straight from SUNRepresentations.jl v0.4.0
# via TensorKitSectors' Fsymbol/Rsymbol -- never hand-authored -- so the Rust
# oracle is a *signed, element-wise* match, i.e. the gauge-continuity claim that
# authorizes a downstream consumer to delete its precomputed SU(3) table.
#
# Scope: every admissible SU(3) sextet (for F) / triple (for R) in which ALL of
# the involved irreps have Weyl dimension <= 27 -- the downstream table's cut.
#
# Run against the session's SUNRepresentations env:
#   julia --project=<sunenv> tools/gen_su3_fr_fixtures.jl <outdir>
# emits <outdir>/su3_fr_f.txt and <outdir>/su3_fr_r.txt.
#
# F line format (whitespace-separated), dynkin as comma-joined "p,q":
#   a b c d e f  mu nu kappa lambda  value       (all 4 mult indices 0-based)
# R line format:
#   a b c  mu nu  value                          (mult indices 0-based)
# The four F multiplicity axes are [mu, nu, kappa, lambda] =
#   [N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}], the TensorKitSectors GenericFusion
# convention. Lines starting with '#' are provenance / comments.

using SUNRepresentations
using Pkg
using Dates
using SHA
const SR = SUNRepresentations
const TKS = SR.TensorKitSectors

outdir = length(ARGS) >= 1 ? ARGS[1] : "tests/fixtures"

weyldim(p, q) = div((p + 1) * (q + 1) * (p + q + 2), 2)
dyn(s) = join(SR.dynkin_label(s), ",")

# All SU(3) irreps with dim <= 27, in a deterministic order.
irreps = SUNIrrep{3}[]
for p in 0:12, q in 0:12
    weyldim(p, q) <= 27 && push!(irreps, SUNIrrep{3}(p, q))
end

function version_of(name)
    for (_, dep) in Pkg.dependencies()
        dep.name == name && return string(dep.version)
    end
    return "unknown"
end

function provenance(io, kind)
    println(io, "# racah SU(3) ", kind, " fixtures -- provenance")
    println(io, "# source: SUNRepresentations.jl v", version_of("SUNRepresentations"),
        " via TensorKitSectors v", version_of("TensorKitSectors"))
    println(io, "# gauge: Gelfand-Tsetlin construction, qrpos!∘cref! (docs/gauge.md)")
    println(io, "# generated (UTC): ", string(Dates.now(Dates.UTC)))
    println(io, "# script sha256: ", bytes2hex(sha256(read(@__FILE__))))
    println(io, "# scope: all admissible SU(3) sextets/triples, every irrep dim <= 27")
    println(io, "# value is Float64; multiplicity indices 0-based")
end

# ---- F symbols ----
open(joinpath(outdir, "su3_fr_f.txt"), "w") do io
    provenance(io, "F-symbol")
    println(io, "# columns: a b c d e f mu nu kappa lambda value")
    nblocks = 0
    nelts = 0
    for a in irreps, b in irreps, c in irreps, d in irreps, e in irreps, f in irreps
        n1 = TKS.Nsymbol(a, b, e)
        n1 > 0 || continue
        n2 = TKS.Nsymbol(e, c, d)
        n2 > 0 || continue
        n3 = TKS.Nsymbol(b, c, f)
        n3 > 0 || continue
        n4 = TKS.Nsymbol(a, f, d)
        n4 > 0 || continue
        F = TKS.Fsymbol(a, b, c, d, e, f)   # size (n1,n2,n3,n4) = (mu,nu,kappa,lambda)
        pre = string(dyn(a), " ", dyn(b), " ", dyn(c), " ", dyn(d), " ", dyn(e), " ", dyn(f))
        for mu in 1:n1, nu in 1:n2, ka in 1:n3, la in 1:n4
            println(io, pre, " ", mu - 1, " ", nu - 1, " ", ka - 1, " ", la - 1,
                " ", repr(F[mu, nu, ka, la]))
            nelts += 1
        end
        nblocks += 1
    end
    println(io, "# emitted ", nblocks, " F blocks, ", nelts, " elements")
end

# ---- R symbols ----
open(joinpath(outdir, "su3_fr_r.txt"), "w") do io
    provenance(io, "R-symbol")
    println(io, "# columns: a b c mu nu value")
    ntriples = 0
    nelts = 0
    for a in irreps, b in irreps, c in irreps
        n = TKS.Nsymbol(a, b, c)
        n > 0 || continue
        R = TKS.Rsymbol(a, b, c)   # n x n
        pre = string(dyn(a), " ", dyn(b), " ", dyn(c))
        for mu in 1:n, nu in 1:n
            println(io, pre, " ", mu - 1, " ", nu - 1, " ", repr(R[mu, nu]))
            nelts += 1
        end
        ntriples += 1
    end
    println(io, "# emitted ", ntriples, " R triples, ", nelts, " elements")
end
