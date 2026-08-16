//! Compatibility-policy and determinism tests for the generated SU(N) authority
//! fingerprint (`racah::sun::sun_authority_fingerprint`, issue #47 leaf L2).
//!
//! Mirrors `tests/su2_fingerprint.rs`, with the generated-family contract: equal
//! bytes identify the convention / generation pipeline / tolerance policy, not
//! byte-identical values (`docs/gauge.md`; issue #47 design record 2).
#![cfg(feature = "cgc-gen")]

use racah::sun::sun_authority_fingerprint;

/// Compatibility policy: the fingerprint's exact current bytes.
///
/// This literal is written out by hand — deliberately NOT derived from
/// `sun_authority_fingerprint()`, so that any change to the returned bytes
/// (a value-affecting convention change or an epoch bump) breaks this assertion
/// and forces a review event.
///
/// Updating this literal REQUIRES a specification correction: the fingerprint is
/// the **version of the SU(N) gauge specification** (`docs/gauge.md`), so it
/// moves only when that document is corrected in a way that changes a returned
/// value, its normalization, or its canonical gauge — never because the code's
/// output drifted. The epoch is per-family: this literal moves independently of
/// `tests/su2_fingerprint.rs` and `tests/bcd_fingerprint.rs`.
///
/// Do not update it to make a test pass. Update it only as one of the four steps
/// of `docs/gauge.md`, "Status": the spec edit, the `epoch=N` bump in
/// `sun_authority_fingerprint`, the CHANGELOG breaking-change entry, and the
/// regenerated `tests/gauge_golden.rs` values — all in the same PR.
#[test]
fn fingerprint_matches_pinned_compatibility_bytes() {
    assert_eq!(
        sun_authority_fingerprint(),
        b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1",
    );
}

/// Determinism: repeated calls return identical bytes and carry no
/// runtime-variant content. Trivially true for a `&'static` byte string; the
/// test documents the contract so a future change that made the value depend on
/// process state (a pointer, a version string, a hash of a mutable input) would
/// break here.
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(sun_authority_fingerprint(), sun_authority_fingerprint());
}
