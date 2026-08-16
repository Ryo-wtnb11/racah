//! Compatibility-policy and determinism tests for the base SU(2) authority
//! fingerprint (`racah::su2_authority_fingerprint`, issue #43 leaf C).

use racah::su2_authority_fingerprint;

/// Compatibility policy: the fingerprint's exact current bytes.
///
/// This literal is written out by hand — it is deliberately NOT derived from
/// `su2_authority_fingerprint()`, so that any change to the returned bytes
/// (including a value-affecting convention change or an epoch bump) breaks this
/// assertion and forces a review event.
///
/// Updating this literal REQUIRES a specification correction: the fingerprint is
/// the **version of the SU(2) convention specification** (`docs/gauge.md` §12),
/// so it moves only when that section is corrected in a way that changes a
/// returned value, its normalization, or its canonical convention — never
/// because the code's output drifted.
///
/// Do not update it to make a test pass. Update it only as one of the four steps
/// of `docs/gauge.md`, "Status": the spec edit, the `epoch=N` bump in
/// `su2_authority_fingerprint`, and the CHANGELOG breaking-change entry.
#[test]
fn fingerprint_matches_pinned_compatibility_bytes() {
    assert_eq!(
        su2_authority_fingerprint(),
        b"racah:su2-exact:model=bigrational-round-once:3j=condon-shortley:cg=condon-shortley:6j=racah-single-sum:f=tks-su2irrep:r=tks-su2irrep:fs=tks-su2irrep:epoch=1",
    );
}

/// Determinism: repeated calls return identical bytes and carry no
/// runtime-variant content. This is trivially true for a `&'static` byte
/// string; the test documents the contract so a future change that made the
/// value depend on process state (a pointer, a version string, a hash of a
/// mutable input) would break here.
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(su2_authority_fingerprint(), su2_authority_fingerprint());
}
