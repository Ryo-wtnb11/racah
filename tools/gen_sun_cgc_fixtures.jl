# Emit SU(N) Clebsch-Gordan coefficient fixtures for the racah gauge-continuity
# oracle (issue #12). Values come straight from SUNRepresentations.jl v0.4.0 --
# never hand-authored -- so the Rust oracle checks a *signed, element-wise*
# match, which is the gauge-continuity claim.
#
# Run against the session's local depot env:
#   julia --project=<scratchpad>/sunenv tools/gen_sun_cgc_fixtures.jl > tests/fixtures/sun_cgc.txt
#
# Line format (whitespace-separated), all indices 1-based as in Julia's
# `basis(s)` = GTPatternIterator order (the Rust reader subtracts 1):
#   N  s1dynkin  s2dynkin  s3dynkin  m1 m2 m3 mu  value
# where each *dynkin is a comma-joined N-1 tuple. Lines starting with '#' are
# provenance / comments.

using SUNRepresentations
using Pkg
using Dates
using SHA
const SR = SUNRepresentations
const NZ = SR.SparseArrayKit.nonzero_pairs

dyn(s) = join(SR.dynkin_label(s), ",")

# Curated (N, s1 dynkin, s2 dynkin) pairs. All product channels of each pair are
# emitted in full; the list is chosen to span N in {2,3,4} and to include
# several outer-multiplicity >= 2 channels (SU(3) 8x8, SU(4) adjoint^2).
pairs = [
    (2, (1,), (1,)),
    (2, (2,), (2,)),
    (2, (3,), (2,)),
    (3, (1, 0), (0, 1)),
    (3, (1, 0), (1, 0)),
    (3, (2, 0), (0, 2)),
    (3, (1, 1), (1, 1)),      # 8 x 8 : the 8 channel has multiplicity 2
    (3, (2, 1), (1, 2)),
    (4, (1, 0, 0), (0, 0, 1)),
    (4, (1, 0, 0), (1, 0, 0)),
    (4, (1, 0, 1), (1, 0, 1)), # adjoint^2 : multiplicity >= 2 channels
]

function version_of(name)
    for (_, dep) in Pkg.dependencies()
        dep.name == name && return string(dep.version)
    end
    return "unknown"
end

println("# racah SU(N) CGC fixtures -- provenance")
println("# source: SUNRepresentations.jl v", version_of("SUNRepresentations"),
        " (Gelfand-Tsetlin construction, qrpos!∘cref! gauge)")
println("# RationalRoots v", version_of("RationalRoots"), " ; julia ", VERSION)
println("# generated (UTC): ", string(Dates.now(Dates.UTC)))
println("# script sha256: ", bytes2hex(sha256(read(@__FILE__))))
println("# indices 1-based (Julia basis order); value is Float64; eltype Float64")
println("# columns: N s1 s2 s3 m1 m2 m3 mu value")

function emit(N, d1, d2)
    s1 = SUNIrrep{N}(d1...)
    s2 = SUNIrrep{N}(d2...)
    for (s3, mult) in SR.directproduct(s1, s2)
        C = CGC(Float64, s1, s2, s3)
        for (k, v) in NZ(C)
            I = Tuple(k)
            println(N, " ", dyn(s1), " ", dyn(s2), " ", dyn(s3), " ",
                    I[1], " ", I[2], " ", I[3], " ", I[4], " ", repr(v))
        end
    end
end

for (N, d1, d2) in pairs
    emit(N, d1, d2)
end
