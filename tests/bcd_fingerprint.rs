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
/// Updating this literal REQUIRES a specification correction: the fingerprint is
/// the **version of the B/C/D gauge specification** (`docs/gauge_soN.md`), so it
/// moves only when that document is corrected in a way that changes a returned
/// value, its normalization, or its canonical gauge — never because the code's
/// output drifted. The epoch is per-family: this literal moves independently of
/// `tests/su2_fingerprint.rs` and `tests/sun_fingerprint.rs`.
///
/// Do not update it to make a test pass. Update it only as one of the four steps
/// of `docs/gauge.md`, "Status": the spec edit, the `epoch=N` bump in
/// `bcd_authority_fingerprint`, the CHANGELOG breaking-change entry, and the
/// regenerated `tests/gauge_golden.rs` values — all in the same PR.
///
/// Epoch history:
/// - `epoch=1` — the initial frozen B/C/D specification.
/// - `epoch=2` — `docs/gauge_soN.md` §14.2 "Base-case frame" (issue #90): the
///   B/D defining seed is re-framed into the sweep's descending-weight order, so
///   every B/D coefficient coupling through the defining rep moves (C values are
///   unchanged — `Setup_SpN` was already in that order).
#[test]
fn fingerprint_matches_pinned_compatibility_bytes() {
    assert_eq!(
        bcd_authority_fingerprint(),
        b"racah:bcd-bootstrap:ref=qspace-v4-dd2cc7e:kron=a-fast:parent=canonical-parent:sweep=gs2-qrpos-posdiag:sort=maxweight-desc:sign=first-significant-positive:align=procrustes-canonical:tol=cg-eps-tier:epoch=2",
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
