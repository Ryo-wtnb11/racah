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
/// Updating this literal REQUIRES a breaking-release decision: the fingerprint
/// changes exactly when a returned SU(2) coefficient value, its normalization,
/// or its canonical convention changes — the same event class the crate's
/// semantic-versioning contract declares breaking (README "Exactness contract",
/// point 4 "Versioned values"; `docs/gauge.md`). Do not update it to make a
/// test pass; update it only as part of that decision, bumping the `epoch=N`
/// tag in `su2_authority_fingerprint` in the same change.
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
