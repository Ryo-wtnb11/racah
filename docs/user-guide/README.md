# racah User Guide

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

Task-oriented documentation: organized by *what you want to compute*, not by
module. It assumes basic representation theory (you know what an irreducible
representation and a tensor product are) and nothing about `racah` internals.

For the exact semantics of one function or type, use the API reference on
[docs.rs/racah](https://docs.rs/racah). For the mathematics, see
[`../theory.pdf`](../theory.pdf). For the normative basis/sign/ordering
conventions of returned values, see [`../gauge.md`](../gauge.md) and
[`../gauge_soN.md`](../gauge_soN.md).

## Chapters

| # | Chapter | Answers |
|---|---|---|
| 1 | [Getting started](getting-started.md) | Install, feature flags, first calculation, how errors work. |
| 2 | [Representations](representations.md) | What an irrep is here, how to build one, Dynkin labels, doubled spins, dimensions, duals, Frobenius–Schur indicators. |
| 3 | [Choosing a group](groups.md) | SU(2) vs SU(N) vs SO(N) vs Spin(N) vs Sp(2N), and the global forms (PSU, SO vs Spin) with admissibility. |
| 4 | [Tensor products and fusion](fusion.md) | Decomposing `a ⊗ b`, reading fusion multiplicities `N^c_ab`. |
| 5 | [Clebsch–Gordan coefficients](clebsch-gordan.md) | What `cgc` returns, index conventions, the multiplicity axis. |
| 6 | [Recoupling: F- and R-symbols](recoupling.md) | What F and R mean, when you need them, block shapes and axis order. |
| 7 | [Numerical behaviour and resources](resources.md) | Exact vs generated values, caches and how to bound them, gauge and reproducibility. |

## Five-minute path

New here? Read [Getting started](getting-started.md), then the chapter for the
object you need. Everything else is reference.

## Worked paths by group

| Group | Start at |
|---|---|
| SU(2) — spins, 3j/6j, CG | [Getting started § SU(2)](getting-started.md#first-calculation-su2-no-features) |
| SU(N) — Dynkin labels, products, CGC | [Representations § SU(N)](representations.md#sun-dynkin-labels) → [Fusion](fusion.md) → [CGC](clebsch-gordan.md) |
| SO(N) — tensor irreps | [Choosing a group § SO(N)](groups.md#son-and-sp2n-the-bcd-series) |
| Spin(N) — spinor irreps | [Choosing a group § Spin(N)](groups.md#spinn-when-you-need-spinors) |
| Sp(2N) | [Choosing a group § Sp(2N) naming](groups.md#sp2n-naming-rank-and-the-defining-dimension) |
| PSU(N), SU(N)/Z_k, SO vs Spin | [Choosing a group § Global forms](groups.md#global-forms-and-admissibility) |
