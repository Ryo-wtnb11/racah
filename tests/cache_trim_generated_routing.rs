//! Isolated process-global generated-tier trim routing contract.
#![cfg(feature = "cgc-gen")]

use racah::bcd::{self, CanonicalCatalog, Series};
use racah::cache::{self, CoefficientCacheTier};
use racah::sun;

fn irrep(d: &[i64]) -> sun::Irrep {
    sun::Irrep::from_dynkin(d).unwrap()
}

#[test]
fn trim_routes_to_each_populated_generated_tier() {
    cache::reset();
    let (a, b, c) = (irrep(&[1, 0]), irrep(&[0, 1]), irrep(&[1, 1]));
    let _ = sun::cgc(&a, &b, &c).unwrap();
    let e8 = irrep(&[1, 1]);
    let _ = sun::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let triv = bcd::Irrep::trivial(Series::C, 2).unwrap();
    let v = bcd::Irrep::from_dynkin(Series::C, &[0, 1]).unwrap();
    let adj = bcd::Irrep::from_dynkin(Series::C, &[2, 0]).unwrap();
    let _ = bcd::f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
    let tiers = [
        CoefficientCacheTier::SunProduct,
        CoefficientCacheTier::SunCgc,
        CoefficientCacheTier::SunF,
        CoefficientCacheTier::BcdCgc,
        CoefficientCacheTier::BcdF,
    ];
    for tier in tiers {
        let before = cache::generated_cache_stats();
        let selected = match tier {
            CoefficientCacheTier::SunProduct => before.sun_product,
            CoefficientCacheTier::SunCgc => before.sun_cgc,
            CoefficientCacheTier::SunF => before.sun_f,
            CoefficientCacheTier::BcdCgc => before.bcd_cgc,
            CoefficientCacheTier::BcdF => before.bcd_f,
            _ => unreachable!(),
        };
        assert!(selected.entries > 0, "fixture must populate {tier}");
        let report = cache::trim_to(tier, 0);
        assert_eq!(report.removed_entries, selected.entries);
        let after = cache::generated_cache_stats();
        let now = match tier {
            CoefficientCacheTier::SunProduct => after.sun_product,
            CoefficientCacheTier::SunCgc => after.sun_cgc,
            CoefficientCacheTier::SunF => after.sun_f,
            CoefficientCacheTier::BcdCgc => after.bcd_cgc,
            CoefficientCacheTier::BcdF => after.bcd_f,
            _ => unreachable!(),
        };
        assert_eq!(now.entries, 0);
    }
}
