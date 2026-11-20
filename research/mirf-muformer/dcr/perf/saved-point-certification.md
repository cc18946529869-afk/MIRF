# Certification of the saved real point estimate

This run uses the accepted 209,644 × 263 point estimate, original alignment and sequence weights, and the fixed published contact groups. It does not call an optimizer. The saved point-estimate files and gauge artifacts are checked against the published reference manifest. The mathematical inputs for the numerical enclosure are the exact reals represented by the saved binary64 inputs and frozen regularization scalars. The objective keeps its original normalization and uses \(W=47899.73496814515\) from the official weight loader.

Each logit is enclosed on a checked dyadic grid. A C accumulator sums integer endpoints exactly after conservative overflow checks. For every row and state, Decimal evaluates the shifted softmax and its weighted residual. One exponential at a directed lower endpoint is enough: for \(0\le d<1\), \(e^d\le (1-d)^{-1}\), which bounds the upper endpoint against the same correctly rounded evaluation. Weighted residual endpoints are rounded outward onto a second dyadic grid, summed by the same checked exact-accumulation principle, then joined to the directed Decimal L2 penalty. The maximum difference from the B binary64 gradient to either enclosure endpoint is a conservative \(\eta\) upper bound, not an estimate of the actual error. The gradient enclosure itself gives an upper bound for the Euclidean parameter radius using the frozen strong-convexity lower bound. The grid interval was checked component by component against the existing direct Decimal implementation on its fixed small fixture. [Python Decimal rounding contract](https://docs.python.org/3.12/library/decimal.html)

The ranking stage verifies that the saved per-site parameters reproduce the source of the published directional artifact. It encloses the real zero-sum gauge, directional average and 20-state Frobenius norm at the saved parameters, and compares with the stored raw score implementation to bound \(\xi\). It then enlarges each raw interval by the two parameter radii, encloses APC over all 263 sites, and applies the five published masks. The report counts certainly-in, certainly-out and unresolved candidates without exporting the underlying rankings. The count bound treats every possible tie conservatively; its integer hit endpoints divided by 263 give the Precision@L interval.

The initial three zero-based sites (0, 131, 262) were chosen before timing, spanning the alignment positions, and all rows were processed for each. The run took 78.45 seconds with three workers (65.35–78.06 seconds per site); process high-water memory was about 600 MiB per worker. The three \(\eta\) upper bounds ranged from \(3.82\times10^{-9}\) to \(5.67\times10^{-9}\), and parameter radii from \(9.66\times10^{-8}\) to \(3.81\times10^{-7}\). The remaining sites used one worker per physical core where possible, with 22 workers and a lower scheduling priority. Progress is written per site to a new scratch directory, including the source hashes.

Run from the repository's `research/mirf-muformer/dcr` directory. Set `SAVED` to the original read-only project directory. Run the first command with the fixed thread environment for B (see [kernel validation](fit-kernel-b-validation.md)). To reproduce exactly the split used in this run, provide both files to the ranking command and prepare an unused progress directory.

```sh
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python3 -m perf.certify_saved_sites --source "$SAVED" --sites 0,131,262 \
  --workers 3 --output /tmp/mirf-site-pilot.json
# Other 260 sites exclude all three pilot IDs.
sites=$(seq 1 261 | awk '$1!=131' | paste -sd, -)
python3 -m perf.certify_saved_sites --source "$SAVED" --sites "$sites" \
  --workers 22 --progress /tmp/mirf-site-progress --output /tmp/mirf-site-remaining.json
python3 -m perf.certify_saved_ranking --source "$SAVED" --workers 16 \
  --site-bounds /tmp/mirf-site-pilot.json /tmp/mirf-site-remaining.json --output /tmp/mirf-ranked.json
```

## Measured result, 2026-11-20

[Three-site pilot](saved-sites-pilot-results.json) · [Remaining 260 site bounds](saved-sites-remaining-results.json) · [Global APC and published-group certificates](saved-top263-results.json)

All **263/263** sites yielded finite parameter radii. The range of \(\eta\) upper bounds was \(3.2811\times10^{-9}\) to \(5.6827\times10^{-9}\) (median \(3.8081\times10^{-9}\)). The parameter radii ranged from \(6.8389\times10^{-8}\) to \(4.6401\times10^{-7}\) (median \(2.8736\times10^{-7}\)). The maximum raw-score \(\xi\) upper bound was \(1.7999\times10^{-16}\). APC itself was evaluated by directed interval arithmetic rather than assigned that raw-score error bound.

| Published group | Numeric boundary gap | Interval slack lower bound (display rounded down) | Guaranteed in Top-263 | Uncertain candidates | Certified contact hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| At least 6 | 2.3310e-4 | 2.3096e-4 | 263 | 0 | 168 |
| At least 12 | 2.2414e-4 | 2.2192e-4 | 263 | 0 | 171 |
| 6–11 | 4.3728e-5 | 4.1279e-5 | 263 | 0 | 48 |
| 12–23 | 1.7953e-4 | 1.7712e-4 | 263 | 0 | 109 |
| At least 24 | 1.4248e-4 | 1.3997e-4 | 263 | 0 | 152 |

**All five published Top-263 sets were strictly certified** for this saved point estimate relative to the exact optimizer of its unchanged objective. Each group reproduced the published numeric score, and all five Precision@L intervals collapsed to the corresponding integer contact count divided by 263. The smallest margin was in the 6–11 group. The conclusion concerns set membership, not the ordering within a selected set. It does not estimate coverage for future simulated alignments.

The 260-site run took 1,204.83 seconds with 22 workers, including the isolated build. Worker time per site was 82.26–127.04 seconds (median 100.38 seconds) with a highest process RSS of 600.80 MiB. The three-site pilot separately took 78.45 seconds. The global stage measured 9.27 seconds for directed raw-score enclosures with 16 workers, then 0.37 seconds for interval APC and group certification. Those last two timings exclude source loading and initial artifact validation. No real parameter optimization, formal simulation, or statistical protocol change was performed.
