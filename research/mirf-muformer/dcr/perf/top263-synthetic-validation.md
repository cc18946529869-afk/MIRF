# Minimal synthetic Top-263 validation

The experiment runs only three fixed numeric cases and reads no research alignment, tree, contact label, or archived parameters. It uses one toy ranking group (all unordered pairs) and ordinary B fits at the existing `1e-5` numerical stopping rule. The dimensions are deliberately small; the 21-state case still has 351 pairs and a nontrivial Top-263 boundary.

The reference imports the stored binary64 values exactly. It encloses the real gradient with directed Decimal arithmetic, including the softmax, weighted accumulation and regularizer. The Decimal exponential is correctly rounded to nearest and then bracketed by adjacent numbers; square-root endpoints are checked by outward multiplication. The gradient interval yields both \(\eta\) against the B output and a direct norm bound for the strong-convexity radius. For fixed parameters, the reference encloses gauge, both directions and the raw Frobenius score and reports \(\xi\) against the binary64 result. It pads the raw bounds by the parameter radii, propagates the full matrix through APC, and compares all selected lower endpoints to all outside upper endpoints. [Python Decimal reference](https://docs.python.org/3.12/library/decimal.html)

| Synthetic case | Sites / states | Max \(\eta\) | Max raw \(\xi\) | Max parameter radius | Certified boundary slack | Top-263 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Grouped, unit weights | 45 / 3 | \(4.41\cdot10^{-15}\) | \(1.15\cdot10^{-16}\) | \(4.37\cdot10^{-5}\) | \(-1.71\cdot10^{-4}\) | No |
| Diffuse, unit weights | 27 / 21 | \(1.48\cdot10^{-15}\) | \(2.49\cdot10^{-16}\) | \(9.97\cdot10^{-5}\) | \(-1.79\cdot10^{-5}\) | No |
| Same diffuse input, objective multiplied by 32 | 27 / 21 | \(3.96\cdot10^{-14}\) | \(3.03\cdot10^{-16}\) | \(4.06\cdot10^{-6}\) | \(+4.28\cdot10^{-4}\) | Yes |

[Machine-readable run](top263_synthetic_results.json) contains the full recorded bounds and wall times. The grouped input had a numerical tie across the boundary. The unit-weight diffuse case had a positive numeric gap of about \(4.44\cdot10^{-4}\) but overlapping intervals. After observing those two rejections, the uniformly scaled case was added as a positive-path control. Multiplying every weight and the penalty total by 32 leaves the exact minimizer unchanged, while the unchanged absolute numerical tolerance yields a tighter certified parameter radius in this control. This is a diagnostic comparison; it does not change the research objective or establish a recipe for formal fits. One of three synthetic cases was certified; this is not an estimate of research-data coverage.

On the server the gradient reference took 1.25–3.74 seconds per case, and gauge/Frobenius/APC and ranking together took 0.11–1.11 seconds. The three phases including fitting took about 12 seconds overall, excluding build/startup. Do not extrapolate these timings to full inputs.

From the repository's `research/mirf-muformer/dcr` directory, using a new output path:

```sh
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 python3 -m perf.check_top263_synthetic --output /tmp/top263-synthetic-new.json
```

The reference is a bounded synthetic validation utility. It does not select formal samples, write checkpoints, compute experimental Precision@L, or certify any research input.
