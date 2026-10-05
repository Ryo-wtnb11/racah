//! A warm F/CGC cache hit hands out the cached coefficients without copying
//! them (issue #118): `FBlock` and `sun::Cgc` share their buffers, so no
//! allocation on a hit is as large as the coefficient buffer. The hit still
//! allocates the owned cache key (one small buffer per label), independent of
//! the coefficient count.

#![cfg(feature = "cgc-gen")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use racah::bcd::{self, CanonicalCatalog, Series};
use racah::sun;

static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        LARGEST.fetch_max(size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `(calls, bytes, largest single allocation)` made by `f`.
fn measure<T>(f: impl FnOnce() -> T) -> (usize, usize, usize, T) {
    let (c0, b0) = (CALLS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    LARGEST.store(0, Ordering::Relaxed);
    let out = f();
    let calls = CALLS.load(Ordering::Relaxed) - c0;
    let bytes = BYTES.load(Ordering::Relaxed) - b0;
    (calls, bytes, LARGEST.load(Ordering::Relaxed), out)
}

/// One test: the counters are process-global, so nothing else may allocate
/// concurrently in this binary.
#[test]
fn warm_hits_do_not_copy_coefficients() {
    // SU(3) 8^6: OM = 2 on every vertex, a 2x2x2x2 block (128 bytes).
    let e8 = sun::Irrep::from_dynkin(&[1, 1]).unwrap();
    let f = || sun::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let cold = f();
    let (calls, bytes, largest, warm) = measure(f);
    let buffer = std::mem::size_of_val(warm.data());
    println!("sun F warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "sun F hit copied its data ({largest} >= {buffer})"
    );

    let c = || sun::cgc(&e8, &e8, &e8).unwrap();
    let cold = c();
    let (calls, bytes, largest, warm) = measure(c);
    let buffer = std::mem::size_of_val(warm.entries());
    println!("sun CGC warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "sun CGC hit copied its entries ({largest} >= {buffer})"
    );

    // Spin(6) adjoint: OM = 2, a 2x2x2x2 block.
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let adj = bcd::Irrep::from_dynkin(Series::D, &[0, 1, 1]).unwrap();
    let mut f = || bcd::f_symbol(&mut cat, &adj, &adj, &adj, &adj, &adj, &adj).unwrap();
    let cold = f();
    let (calls, bytes, largest, warm) = measure(&mut f);
    let buffer = std::mem::size_of_val(warm.data());
    println!("bcd F warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "bcd F hit copied its data ({largest} >= {buffer})"
    );
}
