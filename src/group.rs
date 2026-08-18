//! Root datum plus global form: which dominant weights are genuine
//! representations of *which group* (issue #87, stage (a)).
//!
//! Specifying the Lie **algebra** does not specify the **group**. `so(N)` is
//! the algebra of both `Spin(N)` and `SO(N) = Spin(N)/Z2`; `su(N)` is the
//! algebra of `SU(N)`, of `SU(N)/Z_k` for every `k | N`, and of `PSU(N)`. The
//! groups differ in exactly one respect: **which dominant integral highest
//! weights are genuine group representations**. Fusion rules, CGC, `F`, `R`
//! and the Frobenius–Schur indicator are shared — a global form deletes
//! irreps, it never changes the value of a coefficient of a surviving irrep.
//!
//! # The fact that makes this cheap
//!
//! Let `G_sc` be the simply connected compact group with root system `R`, and
//! `Z = Z(G_sc) ≅ P/Q` (weight lattice modulo root lattice). Every connected
//! compact group with that root system is `G_Γ = G_sc/Γ` for a subgroup
//! `Γ ⊆ Z`, and
//!
//! ```text
//! Irr(G_Γ) = { λ ∈ P⁺ : χ_λ|_Γ = 1 }
//! ```
//!
//! where the central character `χ_λ` depends on `λ` only through its class
//! `[λ] ∈ P/Q`. So a global form is a **per-irrep boolean predicate at
//! construction time** — [`GroupId::admits`](crate::group::GroupId::admits). The class map `P⁺ → P/Q` is a
//! group homomorphism, so `[λ+μ] = [λ]+[μ]`: admissibility is closed under
//! fusion and duality and never needs re-checking on fusion outputs. That is
//! why enforcement is construction-only and why cache keys stay form-free
//! (two forms sharing an irrep *must* share the cache entry).
//!
//! # Conventions
//!
//! Dynkin labels `a = (a₁,…,a_r)`, `aᵢ = ⟨λ, αᵢ^∨⟩`, in Bourbaki numbering —
//! the numbering [`crate::bcd`]'s partition maps already use (see the `bcd`
//! module docs and [`docs/gauge_soN.md`]). External math is pinned to
//! Slansky, *Group Theory for Unified Model Building*, Phys. Rep. **79**
//! (1981) 1–128, §5 (congruency classes, p. 37) and the table notes for
//! Table 36 (SO(8)) and Table 41 (SO(10)); see [`docs/references.md`].
//!
//! Two conventions are *choices*, fixed here once and for all:
//!
//! 1. **`D_r`, `r` even — half-spin forms are named by the class they
//!    RETAIN**, never by the central element they kill and never by an
//!    imported congruence number. Killing `z_{ω_r}` retains `[ω_r]` when
//!    `r ≡ 0 (mod 4)` but retains `[ω_{r-1}]` when `r ≡ 2 (mod 4)`, because
//!    `(ω_r, ω_r) = r/4`; a kill-based name is therefore not stable in `r`.
//!    See [`CenterSubgroup::DHalfSpinPlus`](crate::group::CenterSubgroup::DHalfSpinPlus) / [`CenterSubgroup::DHalfSpinMinus`](crate::group::CenterSubgroup::DHalfSpinMinus).
//! 2. **`D_r`, `r` odd — the `Z4` class uses the generator `c := [ω_r]`**, so
//!    `κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ (mod 4)`, uniformly in `r`.
//!    This agrees with [`crate::bcd::Irrep::dual`] (which flips `λ_r ↦ −λ_r`
//!    for `D_r` odd, i.e. negates `κ`) and with Slansky's SO(10) table note;
//!    it differs from Slansky's single uniform formula by the `Z4`
//!    automorphism `x ↦ −x` for `r ≡ 3 (mod 4)`. That is a generator choice,
//!    not a discrepancy. See [`CenterSubgroup::Z4`](crate::group::CenterSubgroup::Z4).
//!
//! [`docs/gauge_soN.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/gauge_soN.md
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/references.md
//!
//! # Scope
//!
//! Connected compact groups only. `O(N)` and `Pin(N)` are not central
//! quotients of a simply connected group and are out of scope. The `G2`–`E8`
//! variants of [`RootSystem`](crate::group::RootSystem) exist so the type is complete, but carry no
//! predicate logic yet (their centers are `1`, `1`, `Z3`, `Z2`, `1`); only
//! `SimplyConnected` is answerable for them today.

use std::fmt;

/// Root system of a simple compact Lie algebra, by Cartan letter and rank.
///
/// The rank is the number of Dynkin labels. This is a **label-lattice** type:
/// it says which congruences and which weight lattice apply. It is *not* the
/// same job as [`crate::bcd::Series`], which is the **bootstrap-family** key
/// (it indexes the QSpace-provenance defining seeds and the canonical
/// catalog, and exists only where such a seed exists). The bridge is
/// [`crate::bcd::Series::root_system`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RootSystem {
    /// `A_r`, center `Z_{r+1}`. Simply connected form `SU(r+1)`.
    A(usize),
    /// `B_r`, center `Z2`. Simply connected form `Spin(2r+1)`.
    B(usize),
    /// `C_r`, center `Z2`. Simply connected form `Sp(2r)`.
    C(usize),
    /// `D_r`, center `Z2×Z2` (`r` even) or `Z4` (`r` odd). Simply connected
    /// form `Spin(2r)`.
    D(usize),
    /// `G_2`, trivial center.
    G2,
    /// `F_4`, trivial center.
    F4,
    /// `E_6`, center `Z3`.
    E6,
    /// `E_7`, center `Z2`.
    E7,
    /// `E_8`, trivial center.
    E8,
}

impl RootSystem {
    /// The rank — the number of Dynkin labels a weight of this root system
    /// carries.
    pub fn rank(self) -> usize {
        match self {
            RootSystem::A(r) | RootSystem::B(r) | RootSystem::C(r) | RootSystem::D(r) => r,
            RootSystem::G2 => 2,
            RootSystem::F4 => 4,
            RootSystem::E6 => 6,
            RootSystem::E7 => 7,
            RootSystem::E8 => 8,
        }
    }
}

impl fmt::Display for RootSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootSystem::A(r) => write!(f, "A{r}"),
            RootSystem::B(r) => write!(f, "B{r}"),
            RootSystem::C(r) => write!(f, "C{r}"),
            RootSystem::D(r) => write!(f, "D{r}"),
            RootSystem::G2 => write!(f, "G2"),
            RootSystem::F4 => write!(f, "F4"),
            RootSystem::E6 => write!(f, "E6"),
            RootSystem::E7 => write!(f, "E7"),
            RootSystem::E8 => write!(f, "E8"),
        }
    }
}

/// The subgroup `Γ ⊆ Z(G_sc)` that is quotiented out.
///
/// A variant is only meaningful for the root systems named in its docs;
/// [`GroupId::admits`] returns `false` for any other pairing (the named
/// constructors never build one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CenterSubgroup {
    /// `A_r`: the `Z_k ⊆ Z_{r+1}` generated by `ω^{(r+1)/k}`; requires
    /// `k | r+1`. `k = 1` is `SU(r+1)` itself, `k = r+1` is `PSU(r+1)`.
    /// Admissible iff the `N`-ality `Σ i·aᵢ ≡ 0 (mod k)`.
    Zk(usize),
    /// `B_r`, `C_r`, and `D_r` for `r` odd: the unique `Z2 ⊆ Z`.
    ///
    /// - `B_r`: `SO(2r+1)`, admissible iff `a_r` even.
    /// - `C_r`: `PSp(2r)`, admissible iff `a₁+a₃+a₅+… ≡ 0 (mod 2)`.
    /// - `D_r`, `r` odd: `SO(2r)`, admissible iff `a_{r-1}+a_r` even
    ///   (equivalently `κ` even).
    Z2,
    /// `D_r`, `r` odd: the full center `Z4 = ⟨[ω_r]⟩`, giving `PSO(2r)`.
    ///
    /// Admissible iff `κ(λ) ≡ 0 (mod 4)` with
    /// `κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ`.
    ///
    /// Generator convention: `φ(ω_r) = +1`, hence `φ(ω_{r-1}) = −1` and
    /// `φ(ω₁) = 2 = v`. Derivation: `Q` is the even-sum sublattice of `Z^r`
    /// and `4ω_r = (2,…,2) ∈ Q`, so `φ|_{Z^r} = 2Σλᵢ`; substituting
    /// `λᵢ = λ_{i+1}+aᵢ` and using `r` odd gives the formula. Checked against
    /// Slansky Table 41 (SO(10), `D₅`): `45 ↦ 0`, `16 ↦ 1`, `16bar ↦ −1`,
    /// `10 ↦ 2`. Slansky's uniform `D_n` formula equals `+κ` for
    /// `r ≡ 1 (mod 4)` and `−κ` for `r ≡ 3 (mod 4)`; racah pins `φ(ω_r) = +1`
    /// for `r`-uniformity and for agreement with
    /// [`crate::bcd::Irrep::dual`]'s `λ_r ↦ −λ_r`.
    Z4,
    /// `D_r`, `r` even: the index-2 subgroup whose *retained* sublattice is
    /// `{0, v}` — the tensor classes. This is `SO(2r)`.
    ///
    /// Admissible iff `a_{r-1} + a_r ≡ 0 (mod 2)`. This is the only `D`-form
    /// congruence that is `r`-uniform (because `(ε₁,ε₁) = 1` for every `r`),
    /// which is why it is also what [`CenterSubgroup::Z2`] means for `r` odd.
    DVector,
    /// `D_r`, `r` even: the half-spin form that **retains** `[ω_r]`, i.e. the
    /// chirality with `λ_r = +1/2` in the ε-basis.
    ///
    /// Admissible iff `p ≡ 0 (mod 2)` where `p ≡ a_{r-1} + t` and
    /// `t ≡ Σ_{i odd ≤ r-2} aᵢ`. Note the vector rep (`p = q = 1`) is *not*
    /// admitted — a half-spin group has no vector representation.
    ///
    /// **Named by what survives, deliberately.** The kill-based name is not
    /// stable in `r`: `Ann(z_{ω_r}) = {0, s₊}` for `r ≡ 0 (mod 4)` but
    /// `{0, s₋}` for `r ≡ 2 (mod 4)`. Slansky's `Ss(2r)` is a **documented
    /// alias** for this variant (`ω_r ↦ s`, per Table 36 for `D₄`, class
    /// `(1,0)` = sub-`s` = `[ω₄]`, consistent with Table 41 for `D₅`).
    DHalfSpinPlus,
    /// `D_r`, `r` even: the half-spin form that **retains** `[ω_{r-1}]`
    /// (`λ_r = −1/2`).
    ///
    /// Admissible iff `q ≡ 0 (mod 2)` where `q ≡ a_r + t`. Slansky's
    /// `Sc(2r)` is a **documented alias** for this variant (`ω_{r-1} ↦ c`).
    /// See [`CenterSubgroup::DHalfSpinPlus`] for why the naming is by the
    /// retained class.
    DHalfSpinMinus,
    /// `D_r`, `r` even: the full center `Z2×Z2`, giving `PSO(2r)`.
    /// Admissible iff `p ≡ q ≡ 0 (mod 2)`.
    DFull,
}

/// Which connected compact group with a given root system: the cover, or a
/// quotient by a subgroup of its center.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GlobalForm {
    /// The simply connected form `G_sc` — every dominant integral weight is a
    /// representation.
    SimplyConnected,
    /// `G_sc/Γ`.
    Quotient(CenterSubgroup),
}

/// A connected compact simple group, as a root datum: root system plus global
/// form.
///
/// Fields are `pub` so `Ord`/`Hash`/serialization key on the root datum and
/// not on a name — a name is not an identity (`Ss(16) ≅ Sc(16)` as abstract
/// groups; only the retained class distinguishes them). The named
/// constructors below are the ergonomic public surface; they are fallible,
/// matching the house style of [`crate::bcd::Irrep::from_dynkin`].
///
/// # Why the fields stay public (issue #87 §9)
///
/// This is deliberately a **low-level root-datum record**, not a validated
/// handle. Direct construction can pair a [`CenterSubgroup`] with a
/// [`RootSystem`] whose center does not contain it (`A_3` with
/// [`CenterSubgroup::DVector`], `B_2` with `Zk(2)`, any exceptional with any
/// quotient). Such a pairing is not silently wrong and does not propagate:
/// [`admits`](GroupId::admits) returns `false` for **every** weight, and the
/// two form-aware constructors
/// ([`crate::bcd::Irrep::from_dynkin_in`], [`crate::sun::Irrep::from_dynkin_in`])
/// therefore reject every label with the ordinary not-admissible error. A
/// nonsensical datum has no representations, which is the mathematically
/// honest answer for a group that does not exist.
///
/// So no `try_new`, no private fields, no separate raw/validated pair: ordinary
/// users go through [`su`](GroupId::su) / [`so`](GroupId::so) /
/// [`spin`](GroupId::spin) / [`sp`](GroupId::sp) and never touch the root
/// lattice, and the escape hatch is closed by the predicate rather than by
/// encapsulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId {
    /// The root system (and hence the Lie algebra).
    pub root_system: RootSystem,
    /// Which quotient of the simply connected form.
    pub form: GlobalForm,
}

/// Error from a [`GroupId`] constructor. Constructors never panic.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GroupError {
    /// The requested rank or matrix dimension is outside the range for which
    /// the family is defined (e.g. `so(2)`, `sp(3)`, `half_spin_plus(12)`).
    UnsupportedRank {
        /// The constructor that rejected, e.g. `"so"`.
        family: &'static str,
        /// The rejected argument.
        value: usize,
    },
    /// `Z_k` is not a subgroup of the `SU(n)` center: `k` does not divide `n`
    /// (or `k = 0`).
    NotACenterSubgroup {
        /// The `n` of `SU(n)`.
        n: usize,
        /// The requested subgroup order.
        k: usize,
    },
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroupError::UnsupportedRank { family, value } => write!(
                f,
                "{family}({value}) is not a supported rank for this family"
            ),
            GroupError::NotACenterSubgroup { n, k } => write!(
                f,
                "Z{k} is not a subgroup of the SU({n}) center Z{n}: {k} does not divide {n}"
            ),
        }
    }
}

impl std::error::Error for GroupError {}

impl GroupId {
    /// `SU(n)` — `A(n-1)`, simply connected. Requires `n ≥ 2`.
    pub fn su(n: usize) -> Result<Self, GroupError> {
        if n < 2 {
            return Err(GroupError::UnsupportedRank {
                family: "su",
                value: n,
            });
        }
        Ok(GroupId {
            root_system: RootSystem::A(n - 1),
            form: GlobalForm::SimplyConnected,
        })
    }

    /// `SU(n)/Z_k` — requires `n ≥ 2` and `k | n`. `k = 1` is [`Self::su`],
    /// `k = n` is [`Self::psu`].
    pub fn su_quotient(n: usize, k: usize) -> Result<Self, GroupError> {
        if n < 2 {
            return Err(GroupError::UnsupportedRank {
                family: "su",
                value: n,
            });
        }
        if k == 0 || !n.is_multiple_of(k) {
            return Err(GroupError::NotACenterSubgroup { n, k });
        }
        Ok(GroupId {
            root_system: RootSystem::A(n - 1),
            form: GlobalForm::Quotient(CenterSubgroup::Zk(k)),
        })
    }

    /// `PSU(n) = SU(n)/Z_n`, the adjoint form.
    pub fn psu(n: usize) -> Result<Self, GroupError> {
        Self::su_quotient(n, n)
    }

    /// `Sp(m)` — `C(m/2)`, simply connected. `m` is the **matrix dimension**
    /// and must be even and `≥ 2`.
    pub fn sp(m: usize) -> Result<Self, GroupError> {
        Ok(GroupId {
            root_system: RootSystem::C(sp_rank(m)?),
            form: GlobalForm::SimplyConnected,
        })
    }

    /// `PSp(m) = Sp(m)/Z2`, the adjoint form.
    pub fn psp(m: usize) -> Result<Self, GroupError> {
        Ok(GroupId {
            root_system: RootSystem::C(sp_rank(m)?),
            form: GlobalForm::Quotient(CenterSubgroup::Z2),
        })
    }

    /// `Spin(n)` — `B((n-1)/2)` or `D(n/2)` by the parity of `n`, simply
    /// connected. `n` is the **matrix dimension** of the `SO(n)` it covers and
    /// must be `≥ 3`.
    pub fn spin(n: usize) -> Result<Self, GroupError> {
        Ok(GroupId {
            root_system: orthogonal(n, "spin")?,
            form: GlobalForm::SimplyConnected,
        })
    }

    /// `SO(n) = Spin(n)/Z2` — the vector form. `n ≥ 3`.
    pub fn so(n: usize) -> Result<Self, GroupError> {
        let root_system = orthogonal(n, "so")?;
        // For D_r the Z2 is one of three (r even) or the unique one (r odd);
        // either way it is canonically named by the class it retains, so that
        // one group has one `GroupId`. `Z2` stays accepted by `admits` for D.
        let sub = match root_system {
            RootSystem::D(_) => CenterSubgroup::DVector,
            _ => CenterSubgroup::Z2,
        };
        Ok(GroupId {
            root_system,
            form: GlobalForm::Quotient(sub),
        })
    }

    /// `PSO(n)`, the adjoint form. For `n` odd this *is* `SO(n)` (the `B_r`
    /// center is already quotiented out); for `n = 2r` it is the quotient by
    /// the full center.
    pub fn pso(n: usize) -> Result<Self, GroupError> {
        let root_system = orthogonal(n, "pso")?;
        let sub = match root_system {
            RootSystem::B(_) => CenterSubgroup::Z2,
            RootSystem::D(r) if r % 2 == 0 => CenterSubgroup::DFull,
            _ => CenterSubgroup::Z4,
        };
        Ok(GroupId {
            root_system,
            form: GlobalForm::Quotient(sub),
        })
    }

    /// The half-spin form of `Spin(n)` that **retains** the `[ω_r]` chirality
    /// (Slansky's `Ss(n)`). Requires `n = 2r` with `r` even, i.e.
    /// `n ≡ 0 (mod 4)`, `n ≥ 8`.
    pub fn half_spin_plus(n: usize) -> Result<Self, GroupError> {
        Ok(GroupId {
            root_system: half_spin_root_system(n)?,
            form: GlobalForm::Quotient(CenterSubgroup::DHalfSpinPlus),
        })
    }

    /// The half-spin form of `Spin(n)` that **retains** the `[ω_{r-1}]`
    /// chirality (Slansky's `Sc(n)`). Requires `n ≡ 0 (mod 4)`, `n ≥ 8`.
    pub fn half_spin_minus(n: usize) -> Result<Self, GroupError> {
        Ok(GroupId {
            root_system: half_spin_root_system(n)?,
            form: GlobalForm::Quotient(CenterSubgroup::DHalfSpinMinus),
        })
    }

    /// Whether `dynkin` is a genuine representation of this group.
    ///
    /// `dynkin` is the dominant integral weight in Bourbaki numbering. Returns
    /// `false` for a label of the wrong length, a label with a negative
    /// component, and for a [`CenterSubgroup`] that is not a subgroup of this
    /// root system's center (the named constructors never build one).
    ///
    /// Admissibility is closed under fusion and duality, so this only needs
    /// calling at construction.
    pub fn admits(&self, dynkin: &[i64]) -> bool {
        let r = self.root_system.rank();
        if dynkin.len() != r || dynkin.iter().any(|&a| a < 0) {
            return false;
        }
        let sub = match self.form {
            // A global form never introduces a weight the cover lacks, and the
            // cover admits every dominant integral weight.
            GlobalForm::SimplyConnected => return true,
            GlobalForm::Quotient(sub) => sub,
        };
        match (self.root_system, sub) {
            // A_r: N-ality Σ i·aᵢ mod k. k = 1 is SU(N) itself.
            (RootSystem::A(_), CenterSubgroup::Zk(k)) => {
                if k == 0 || !(r + 1).is_multiple_of(k) {
                    return false;
                }
                let k = k as i64;
                let nality: i64 = dynkin
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| (i as i64 + 1) * a)
                    .sum();
                nality.rem_euclid(k) == 0
            }
            // B_r: [ω_i] = 0 for i < r, [ω_r] = z. SO(2r+1).
            (RootSystem::B(_), CenterSubgroup::Z2) => dynkin[r - 1] % 2 == 0,
            // C_r: [ω_i] = (i mod 2)·z. PSp(2r).
            (RootSystem::C(_), CenterSubgroup::Z2) => odd_index_sum(dynkin) % 2 == 0,
            // D_r: the vector (tensor) sublattice {0, v}. SO(2r), r-uniform.
            (RootSystem::D(_), CenterSubgroup::Z2 | CenterSubgroup::DVector) if r >= 2 => {
                (dynkin[r - 2] + dynkin[r - 1]) % 2 == 0
            }
            // D_r, r odd: PSO(2r) is the quotient by the full Z4.
            (RootSystem::D(_), CenterSubgroup::Z4) if r >= 3 && r % 2 == 1 => {
                d_odd_class(dynkin) == 0
            }
            // D_r, r even: the two half-spin forms and PSO(2r).
            (RootSystem::D(_), sub) if r >= 2 && r.is_multiple_of(2) => {
                let t = odd_index_sum(&dynkin[..r - 2]);
                let p = (dynkin[r - 2] + t) % 2;
                let q = (dynkin[r - 1] + t) % 2;
                match sub {
                    CenterSubgroup::DHalfSpinPlus => p == 0,
                    CenterSubgroup::DHalfSpinMinus => q == 0,
                    CenterSubgroup::DFull => p == 0 && q == 0,
                    _ => false,
                }
            }
            // Not a subgroup of this root system's center (or an exceptional
            // family, whose quotient forms carry no logic yet).
            _ => false,
        }
    }
}

/// `Σ_{i odd, 1-based} aᵢ` — the `i = 1, 3, 5, …` entries of `a`.
fn odd_index_sum(dynkin: &[i64]) -> i64 {
    dynkin.iter().step_by(2).sum()
}

/// The `Z4` central class `κ(λ) ∈ {0,1,2,3}` of a `D_r` weight with `r` odd,
/// under the generator `c = [ω_r]`:
///
/// ```text
/// κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ  (mod 4)
/// ```
///
/// Exposed because this is the value an external congruency table is checked
/// against; see [`CenterSubgroup::Z4`] for the derivation and the Slansky
/// SO(10) anchors. Returns `0` for a label shorter than 2.
pub fn d_odd_central_class(dynkin: &[i64]) -> i64 {
    if dynkin.len() < 2 {
        return 0;
    }
    d_odd_class(dynkin)
}

/// `κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ (mod 4)` for `D_r`, `r` odd,
/// with the generator `c = [ω_r]` (see [`CenterSubgroup::Z4`]).
fn d_odd_class(dynkin: &[i64]) -> i64 {
    let r = dynkin.len();
    (dynkin[r - 1] - dynkin[r - 2] + 2 * odd_index_sum(&dynkin[..r - 2])).rem_euclid(4)
}

fn sp_rank(m: usize) -> Result<usize, GroupError> {
    if m < 2 || !m.is_multiple_of(2) {
        return Err(GroupError::UnsupportedRank {
            family: "sp",
            value: m,
        });
    }
    Ok(m / 2)
}

/// `B((n-1)/2)` or `D(n/2)`, from the matrix dimension `n ≥ 3`.
fn orthogonal(n: usize, family: &'static str) -> Result<RootSystem, GroupError> {
    match n {
        n if n < 3 => Err(GroupError::UnsupportedRank { family, value: n }),
        n if n % 2 == 1 => Ok(RootSystem::B((n - 1) / 2)),
        n => Ok(RootSystem::D(n / 2)),
    }
}

fn half_spin_root_system(n: usize) -> Result<RootSystem, GroupError> {
    // n = 2r with r even ⇔ n ≡ 0 (mod 4); r ≥ 4 keeps D_r semisimple and the
    // two chiralities distinct.
    if n < 8 || !n.is_multiple_of(4) {
        return Err(GroupError::UnsupportedRank {
            family: "half_spin",
            value: n,
        });
    }
    Ok(RootSystem::D(n / 2))
}
