"""Synthetic checks of the B fitting interface. No research data is loaded."""

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import potts_fit
from perf import bench_layout, fit_kernel_b


def fixture(q, length, rows=11):
    rng = np.random.default_rng(20261120 + q + length)
    msa = rng.integers(0, q, size=(rows, length), dtype=np.int8)
    msa[0] = 0
    msa[1] = q - 1
    weights = rng.uniform(0.1, 2.0, size=rows)
    weights[0] = 0.0
    count = potts_fit.parameter_count(length, q)
    return msa, weights, rng.normal(0, 0.17, count), rng.normal(0, 0.11, count)


def fingerprint(bound):
    sha = hashlib.sha256()
    library = bound.library_for_worker()
    msa, weights, theta, direction = fixture(21, 263)
    total = float(weights.sum())
    for site in (0, 131, 262):
        nll, penalty, gradient = potts_fit.site_terms_c(
            theta, msa, weights, site, total, q=21, library=library
        )
        prob, velocity = potts_fit._direction_cache(theta, direction, msa, site, 21, library=library)
        delta = potts_fit.make_direct_delta(msa, weights, site, total, q=21, library=library)
        for value in (nll, penalty, gradient, prob, velocity, delta(theta, direction, 0.31)):
            sha.update(np.asarray(value, dtype="<f8").tobytes())
    return sha.hexdigest()


class BInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        tmp = Path(cls.temp.name)
        cls.artifacts = tmp / "b"
        cls.manifest = fit_kernel_b.build(cls.artifacts)
        cls.expected_sha = cls.manifest["artifact_sha256"]
        cls.bound = fit_kernel_b.BoundBKernel(cls.artifacts, expected_sha256=cls.expected_sha)
        cls.b = cls.bound.library_for_worker()
        cls.a_path = tmp / "libA.so"
        bench_layout.compile_library(bench_layout.SOURCE_A, cls.a_path, native=False)
        cls.a = ctypes.CDLL(str(cls.a_path))
        for name in fit_kernel_b.FIT_SYMBOLS:
            a_symbol = getattr(cls.a, name)
            b_symbol = getattr(cls.b, name)
            a_symbol.argtypes, a_symbol.restype = b_symbol.argtypes, b_symbol.restype

    def test_a_b_agreement_at_same_parameters(self):
        maxima = dict(objective=0.0, gradient=0.0, probability=0.0, velocity=0.0, direct_delta=0.0)
        for q, length in ((3, 8), (21, 263)):
            msa, weights, theta, direction = fixture(q, length)
            total = float(weights.sum())
            for site in (0, length // 2, length - 1):
                def terms(x, library):
                    return potts_fit.site_terms_c(x, msa, weights, site, total, q=q, library=library)

                left, right = terms(theta, self.a), terms(theta, self.b)
                maxima["objective"] = max(maxima["objective"], abs(sum(left[:2]) - sum(right[:2])))
                maxima["gradient"] = max(maxima["gradient"], float(np.max(np.abs(left[2]-right[2]))))
                np.testing.assert_allclose(left[:2], right[:2], rtol=1e-13, atol=1e-12)
                np.testing.assert_allclose(left[2], right[2], rtol=1e-13, atol=1e-12)
                caches = [
                    potts_fit._direction_cache(theta, direction, msa, site, q, library=lib)
                    for lib in (self.a, self.b)
                ]
                for name, a, b in zip(("probability", "velocity"), *caches):
                    maxima[name] = max(maxima[name], float(np.max(np.abs(a-b))))
                    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)
                changes = [
                    potts_fit.make_direct_delta(msa, weights, site, total, q=q, library=lib)
                    for lib in (self.a, self.b)
                ]
                for alpha in (0.0, 1e-8, -0.25, 0.7, 4.0):
                    x, y = (fn(theta, direction, alpha) for fn in changes)
                    maxima["direct_delta"] = max(maxima["direct_delta"], abs(x-y))
                    self.assertAlmostEqual(x, y, delta=1e-11)
                    self.assertAlmostEqual(y, sum(terms(theta + alpha * direction, self.b)[:2]) - sum(right[:2]), delta=1e-10)
        self.assertEqual(self.bound.objective_team_size(), 1)
        print("Maximum absolute A/B differences:", maxima)

    def test_fit_site_b_routes_all_calls_without_touching_a(self):
        msa, weights, theta, direction = fixture(3, 8)
        total = float(weights.sum())
        reference = potts_fit.site_terms_c(theta, msa, weights, 2, total, q=3, library=self.b)
        delta = potts_fit.make_direct_delta(msa, weights, 2, total, q=3, library=self.b)
        previous = potts_fit._COMPILED

        def check_route(objective, initial, *, delta_fn, **kwargs):
            np.testing.assert_array_equal(initial, np.zeros_like(theta))
            value, gradient = objective(theta)
            self.assertEqual(value, sum(reference[:2]))
            np.testing.assert_array_equal(gradient, reference[2])
            self.assertEqual(delta_fn(theta, direction, 0.4), delta(theta, direction, 0.4))
            return potts_fit.OptimizerResult("interface-check", theta, value, float(np.max(np.abs(gradient))), 0, 1, "no optimizer run")

        with mock.patch.object(potts_fit, "_compile_library", side_effect=AssertionError("B fell back to A")):
            with mock.patch.object(potts_fit, "lbfgs", side_effect=check_route):
                result = potts_fit.fit_site(msa, weights, 2, total, q=3, b_kernel=self.bound)
        self.assertEqual(result.status, "interface-check")
        self.assertIs(potts_fit._COMPILED, previous)


    def test_real_lbfgs_a_and_b_end_to_end(self):
        for q, length, site in ((3, 8, 2), (21, 263, 131)):
            with self.subTest(states=q, columns=length, site=site):
                msa, weights, _, _ = fixture(q, length, rows=31)
                total = float(weights.sum())
                with mock.patch.object(potts_fit, "_LIBRARY", self.a_path), mock.patch.object(potts_fit, "_COMPILED", self.a):
                    a = potts_fit.fit_site(msa, weights, site, total, q=q)
                with mock.patch.object(potts_fit, "_compile_library", side_effect=AssertionError("B fell back to A")):
                    b = potts_fit.fit_site(msa, weights, site, total, q=q, b_kernel=self.bound)
                g_a = potts_fit.site_terms_c(a.theta, msa, weights, site, total, q=q, library=self.a)[2]
                g_b = potts_fit.site_terms_c(b.theta, msa, weights, site, total, q=q, library=self.b)[2]
                self.assertEqual(a.status, "converged")
                self.assertEqual(a.status, b.status)
                self.assertGreater(a.n_strong_wolfe, 0)
                self.assertLessEqual(float(np.max(np.abs(g_b))), potts_fit.OPT_TOL)
                for name in ("objective", "objective_accumulated", "grad_inf", "iterations", "evaluations",
                             "n_strong_wolfe", "n_armijo_fallback", "n_curvature_skipped", "trace"):
                    with self.subTest(field=name):
                        self.assertEqual(getattr(a, name), getattr(b, name))
                np.testing.assert_array_equal(a.theta, b.theta)
                np.testing.assert_array_equal(g_a, g_b)
                print("Real L-BFGS A/B:", q, length, site, b.iterations, b.evaluations,
                      b.n_strong_wolfe, b.n_armijo_fallback, b.grad_inf, len(b.trace))

    def test_exact_symbol_provider_and_mismatched_identity(self):
        expected = str(self.artifacts / fit_kernel_b.LIBRARY)
        self.assertEqual({self.bound.symbol_paths[name] for name in fit_kernel_b.FIT_SYMBOLS}, {expected})
        with self.assertRaisesRegex(RuntimeError, "hash"):
            fit_kernel_b.BoundBKernel(self.artifacts, expected_sha256="0" * 64)
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with self.assertRaisesRegex(RuntimeError, "OMP_NUM_THREADS"):
                self.bound.library_for_worker()
        invalid = Path(self.temp.name) / "wrong-symbols"
        shutil.copytree(self.artifacts, invalid)
        artifact = invalid / fit_kernel_b.LIBRARY
        artifact.unlink()
        # The old B benchmark lacks the direct-delta ABI, despite a valid ELF/hash.
        bench_layout.compile_library(bench_layout.SOURCE_T, artifact, native=False)
        artifact.chmod(0o444)
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = invalid / fit_kernel_b.MANIFEST
        manifest.chmod(0o644)
        import json
        manifest.write_text(json.dumps({**self.manifest, "artifact_sha256": sha}))
        with self.assertRaisesRegex(AttributeError, "site_delta_nll"):
            fit_kernel_b.BoundBKernel(invalid, expected_sha256=sha)

    def test_fingerprints_repeat_in_fresh_processes(self):
        script = (
            "from pathlib import Path; from perf.fit_kernel_b import BoundBKernel; "
            "from test_fit_kernel_b import fingerprint; "
            f"print(fingerprint(BoundBKernel(Path({str(self.artifacts)!r}), expected_sha256={self.expected_sha!r})))"
        )
        expected = fingerprint(self.bound)
        self.assertEqual(fingerprint(self.bound), expected)
        for _ in range(2):
            actual = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
            self.assertEqual(actual, expected)
        print("Synthetic B fingerprint:", expected)
        print("B artifact SHA-256:", self.expected_sha)
        print("B compiler:", self.manifest["compiler_version"])

    def test_guard_rejects_an_actual_two_thread_team(self):
        script = (
            "import ctypes; import potts_fit; from perf.fit_kernel_b import _bind; "
            "from test_fit_kernel_b import fixture; "
            f"lib=ctypes.CDLL({str(self.artifacts / fit_kernel_b.LIBRARY)!r}); _bind(lib); "
            "lib.omp_set_dynamic(0); lib.omp_set_num_threads(2); "
            "msa,w,theta,_=fixture(3,8); "
            "\ntry: potts_fit.site_terms_c(theta,msa,w,0,float(w.sum()),q=3,library=lib)\n"
            "except RuntimeError as error:\n"
            "    assert 'one OpenMP thread' in str(error) and lib.site_objective_team_size() == 2\n"
            "else: raise AssertionError('two-thread B team accepted')"
        )
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2"}
        subprocess.run([sys.executable, "-c", script], env=env, check=True)
