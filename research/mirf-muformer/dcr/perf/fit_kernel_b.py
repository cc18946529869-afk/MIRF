"""Build and load the opt-in, single-thread Potts B fitting artifact on Linux."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .bench_layout import BASE_FLAGS, SOURCE_T, bind

SOURCES = (SOURCE_T, SOURCE_T.with_name("conditional_delta.c"))
LIBRARY = "libpotts_b_fit.so"
MANIFEST = "build.json"
FIT_SYMBOLS = ("site_objective", "site_direction_cache", "site_delta_nll")
RUNTIME_SYMBOLS = {"libgomp.so.1": "omp_set_num_threads", "libm.so.6": "exp", "libc.so.6": "malloc"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The B fitting artifact requires Linux ELF mappings")


def _host() -> dict:
    cpu_fields = {"vendor_id", "model name", "flags", "Features"}
    cpu = set()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() in cpu_fields:
            cpu.add(f"{key.strip()}:{value.strip()}")
    return {
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu": sorted(cpu),
        "python": sys.version,
        "numpy": np.__version__,
    }


def _tool(name: str, path: str) -> Path:
    resolved = shutil.which(name, path=path)
    if resolved is None:
        raise RuntimeError(f"Build tool not found: {name}")
    return Path(resolved).resolve(strict=True)


def build(output: Path) -> dict:
    """Write a new binary and manifest; never replace an existing artifact."""
    _require_linux()
    compiler = _tool(BASE_FLAGS[0], os.environ.get("PATH", os.defpath))
    build_env = {"PATH": f"{compiler.parent}:{os.defpath}", "LC_ALL": "C"}
    tools = {"gcc": _identity(compiler)}
    for unit in ("cc1", "as", "ld"):
        name = subprocess.check_output(
            [str(compiler), f"-print-prog-name={unit}"], env=build_env, text=True
        ).strip()
        tools[unit] = _identity(_tool(name, build_env["PATH"]))
    compiler_version = subprocess.check_output(
        [str(compiler), "--version"], env=build_env, text=True
    ).splitlines()[0]

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = output / LIBRARY
    command = [
        str(compiler), *BASE_FLAGS[1:], "-DMIRF_B_FIT_SINGLE_THREAD",
        *(str(source) for source in SOURCES), "-o", str(library), "-lm",
    ]
    subprocess.run(command, check=True, env=build_env)
    library.chmod(0o444)

    dependencies = {}
    ldd = subprocess.check_output(
        [str(_tool("ldd", build_env["PATH"])), str(library)], env=build_env, text=True
    )
    for line in ldd.splitlines():
        soname, arrow, location = line.strip().partition(" => ")
        if arrow and soname in RUNTIME_SYMBOLS:
            dependencies[soname] = _identity(Path(location.split(" (", 1)[0]))
    if dependencies.keys() != RUNTIME_SYMBOLS.keys():
        raise RuntimeError("Could not identify B runtime dependencies")

    manifest = {
        "format": 1,
        "artifact_sha256": _sha256(library),
        "command": command,
        "compiler_version": compiler_version,
        "build_environment": build_env,
        "tools": tools,
        "sources": {source.name: _sha256(source) for source in SOURCES},
        "runtime_dependencies": dependencies,
        "host": _host(),
        "thread_environment": THREAD_ENV,
    }
    path = output / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    return manifest


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def _loaded_object(symbol) -> Path:
    address = ctypes.cast(symbol, ctypes.c_void_p)
    info = _DlInfo()
    lookup = ctypes.CDLL(None).dladdr
    lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    lookup.restype = ctypes.c_int
    if not lookup(address, ctypes.byref(info)) or not info.filename:
        raise RuntimeError(f"Cannot identify provider of {symbol.__name__}")
    path = Path(os.fsdecode(info.filename)).resolve(strict=True)
    identity = path.stat()
    # The mapped ELF identity must still match the file we hash by path.
    for line in Path("/proc/self/maps").read_text().splitlines():
        extent, perms, _, device, inode, *_ = line.split(maxsplit=5)
        lower, upper = (int(limit, 16) for limit in extent.split("-"))
        if lower <= address.value < upper:
            major, minor = (int(part, 16) for part in device.split(":"))
            if (
                "x" in perms
                and (major, minor) == (os.major(identity.st_dev), os.minor(identity.st_dev))
                and int(inode) == identity.st_ino
            ):
                return path
            break
    raise RuntimeError(f"Loaded mapping differs from file for {symbol.__name__}")


def _check_thread_environment() -> None:
    for name, value in THREAD_ENV.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Set {name}={value} before starting the B worker")


def _bind(library: ctypes.CDLL) -> None:
    bind(library)
    double_p = ctypes.POINTER(ctypes.c_double)
    library.site_delta_nll.argtypes = [
        double_p, double_p, ctypes.POINTER(ctypes.c_int8), double_p,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, double_p,
    ]
    library.site_delta_nll.restype = ctypes.c_int
    for symbol in (library.site_objective_team_size, library.site_fit_fp_ready):
        symbol.argtypes = []
        symbol.restype = ctypes.c_int


class BoundBKernel:
    """Identity-checked B library; every fitting thread must activate it."""

    def __init__(self, directory: Path, *, expected_sha256: str):
        _require_linux()
        _check_thread_environment()
        directory = directory.resolve(strict=True)
        self.manifest = json.loads((directory / MANIFEST).read_text())
        path = directory / LIBRARY
        if (
            self.manifest["format"] != 1
            or self.manifest["artifact_sha256"] != expected_sha256
            or _sha256(path) != expected_sha256
            or path.stat().st_mode & 0o222
        ):
            raise RuntimeError("B library hash or read-only artifact check failed")
        if (
            self.manifest["host"] != _host()
            or self.manifest["thread_environment"] != THREAD_ENV
            or self.manifest["sources"] != {source.name: _sha256(source) for source in SOURCES}
        ):
            raise RuntimeError("B host, thread configuration, or source identity differs from build")

        self._library = ctypes.CDLL(str(path))
        _bind(self._library)
        self.symbol_paths = {}
        for symbol in FIT_SYMBOLS + ("site_objective_team_size", "site_fit_fp_ready"):
            provider = _loaded_object(getattr(self._library, symbol))
            if provider != path or _sha256(provider) != expected_sha256:
                raise RuntimeError(f"{symbol} is not from the frozen B library")
            self.symbol_paths[symbol] = str(provider)

        for soname, symbol in RUNTIME_SYMBOLS.items():
            provider = _loaded_object(getattr(self._library, symbol))
            if _identity(provider) != self.manifest["runtime_dependencies"][soname]:
                raise RuntimeError(f"B runtime dependency changed: {soname}")
        self.library_for_worker()

    def library_for_worker(self) -> ctypes.CDLL:
        _check_thread_environment()
        for name, value in (("omp_set_dynamic", 0), ("omp_set_num_threads", 1)):
            setter = getattr(self._library, name)
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            setter(value)
        if self._library.site_fit_fp_ready() != 1:
            raise RuntimeError("B fitting requires round-to-nearest and gradual underflow")
        return self._library

    def objective_team_size(self) -> int:
        """Report the team measured inside the latest objective in this thread."""
        return int(self._library.site_objective_team_size())
