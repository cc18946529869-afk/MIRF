# Opt-in B fitting kernel

This implementation branch closes the missing C entry point for interface review. It does not adopt B for the research runs. A remains the default in `fit_site`; B is selected only through `b_kernel=BoundBKernel(...)`. No optimizer, acceptance threshold, objective, or line-search arithmetic is changed.

The fitting artifact links `conditional_transpose.c` and `conditional_delta.c` with the baseline `BASE_FLAGS` and the single-thread guard. The body of `site_delta_nll` is byte-for-byte the body from A. The benchmark-only build remains available with its original flags. The B loader checks the expected SHA-256, read-only binary, build and host manifest, and the actual executable mapping of all three function pointers. It also checks the providers for the math, C, and OpenMP runtimes recorded when building. The worker sets the effective OpenMP policy, and the B objective records and enforces the team size inside its parallel region. This is a Linux/GCC/glibc interface.

Run from the repository's `research/mirf-muformer/dcr` directory:

```sh
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_fit_kernel_b
```

For an authorized caller, `perf.fit_kernel_b.build(new_directory)` writes a new `libpotts_b_fit.so` and `build.json`. Pass the build's separately recorded artifact SHA-256 to `BoundBKernel(new_directory, expected_sha256=...)`. Keep the binary and manifest out of Git. Use a separate artifact for each build or environment.

2026-11-20 server verification, GCC 13.3.0 and NumPy 1.26.4:

- The five interface tests passed. In two synthetic input shapes (3 states and 8 sites; 21 states and 263 sites), with the same parameters, weights and sites, the maximum absolute A/B difference observed was 0 for the objective, gradient, cached probabilities, cached direction logits and full direct objective difference. The direct increment was also checked against the objective change at five step sizes.
- The three actual symbols mapped to the same B artifact. A mismatched expected hash and the earlier two-entry benchmark library were rejected. The in-region probe saw one thread on the accepted path and rejected an actual two-thread team.
- The output fingerprint matched within one worker and across two fresh processes: `28ff8bf2efabb3dba4ae0f9118efe5595c59ea7f6a512c222e278086680543c8`. This test build's binary SHA-256 was `bac79187e96039fabdd603360148d692da097e34f774c7b84377d34b4ba6dc4b`.
- All 36 existing `test_potts` tests passed with A compiled into a temporary directory. The B wiring test substitutes an interface probe for L-BFGS, so it does not optimize a site.

The interface results apply to these fixtures and this build. The historical research assets and the production run approval remain separate.
