//! Compatibility-policy and determinism tests for the generated SO(N)/Sp(2N)
//! authority fingerprint (`racah::bcd::bcd_authority_fingerprint`, issue #47
//! leaf L2).
//!
//! Mirrors `tests/su2_fingerprint.rs`, with the generated-family contract: equal
//! bytes identify the convention / generation pipeline / tolerance policy, not
//! byte-identical values (`docs/gauge_soN.md`; issue #47 design record 2).
#![cfg(feature = "cgc-gen")]

use racah::bcd::bcd_authority_fingerprint;

/// Compatibility policy: the fingerprint's exact current bytes.
///
/// This literal is written out by hand — deliberately NOT derived from
/// `bcd_authority_fingerprint()`, so that any change to the returned bytes
/// (a value-affecting convention change or an epoch bump) breaks this assertion
/// and forces a review event.
///
/// Updating this literal REQUIRES a breaking-release decision: the B/C/D
/// fingerprint changes exactly when a returned SO(N)/Sp(2N) coefficient value,
/// its normalization, or its canonical gauge changes — the breaking-release
/// event class of `docs/gauge_soN.md`. The epoch is per-family: this literal
/// moves independently of `tests/su2_fingerprint.rs` and
/// `tests/sun_fingerprint.rs`. Do not update it to make a test pass; update it
/// only as part of that decision, bumping the `epoch=N` tag in
/// `bcd_authority_fingerprint` in the same change.
#[test]
fn fingerprint_matches_pinned_compatibility_bytes() {
    assert_eq!(
        bcd_authority_fingerprint(),
        b"racah:bcd-bootstrap:ref=qspace-v4-dd2cc7e:kron=a-fast:parent=canonical-parent:sweep=gs2-qrpos-posdiag:sort=maxweight-desc:sign=first-significant-positive:align=procrustes-canonical:tol=cg-eps-tier:epoch=1",
    );
}

/// Determinism: repeated calls return identical bytes and carry no
/// runtime-variant content. Trivially true for a `&'static` byte string; the
/// test documents the contract so a future change that made the value depend on
/// process state (a pointer, a version string, a hash of a mutable input) would
/// break here.
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(bcd_authority_fingerprint(), bcd_authority_fingerprint());
}
