"""Read a ProteinGym A2M alignment and cut the mature 263 columns.

In this file every column is one fixed target position. Uppercase and lowercase
letters are both residues; case marks coverage, not an insertion. '-' and '.'
are gaps. No column is dropped because it is lowercase.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP_CODE = 0
CODE = {amino: index + 1 for index, amino in enumerate(AMINO_ACIDS)}
TARGET_SEQUENCE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTF"
    "KVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTI"
    "GGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWME"
    "ADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGA"
    "SLIKHW"
)
OFFICIAL_HEADER = "BLAT_ECOLX/1-286"
MATURE_START = 24
MATURE_END = 286


class MsaFormatError(ValueError):
    """The alignment cannot be converted without guessing a symbol."""


@dataclass(frozen=True)
class MsaAudit:
    n_sequences: int
    n_alignment_columns: int
    n_uppercase_columns: int
    n_lowercase_columns: int
    mature_lowercase_positions: tuple[int, ...]
    nonstandard_mapped_to_gap: dict[str, int]
    mature_gap_fraction: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "n_sequences": self.n_sequences,
            "n_alignment_columns": self.n_alignment_columns,
            "n_uppercase_columns": self.n_uppercase_columns,
            "n_lowercase_columns": self.n_lowercase_columns,
            "mature_lowercase_positions": self.mature_lowercase_positions,
            "nonstandard_mapped_to_gap": dict(self.nonstandard_mapped_to_gap),
            "mature_gap_fraction": self.mature_gap_fraction.copy(),
        }


def read_a2m(text: str) -> list[tuple[str, str]]:
    """Parse FASTA/A2M text. Every sequence must have the same length."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        records.append((header, "".join(chunks)))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
            continue
        if header is None:
            raise MsaFormatError("sequence text appears before the first header")
        chunks.append("".join(line.split()))
    flush()
    if not records:
        raise MsaFormatError("alignment has no sequences")
    width = len(records[0][1])
    for name, sequence in records:
        if len(sequence) != width:
            raise MsaFormatError(
                f"{name} has length {len(sequence)}; the first sequence has {width}"
            )
    return records


def is_lowercase_letter(symbol: str) -> bool:
    """True for any one-character lowercase letter, not only the 20 amino acids."""
    return len(symbol) == 1 and symbol.isalpha() and symbol.islower()


def encode_symbol(symbol: str, nonstandard: Counter[str]) -> int:
    """Map one alignment character. Lowercase standard residues stay residues."""
    if symbol in ("-", "."):
        return GAP_CODE
    normalized = symbol.upper()
    code = CODE.get(normalized)
    if code is None:
        nonstandard[symbol] += 1
        return GAP_CODE
    return code


def encode_fixed_columns(
    records: list[tuple[str, str]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode every column. Nothing is removed because of letter case."""
    width = len(records[0][1])
    encoded = np.empty((len(records), width), dtype=np.int8)
    nonstandard: Counter[str] = Counter()
    for row, (_, sequence) in enumerate(records):
        if len(sequence) != width:
            raise MsaFormatError(
                f"row {row + 1} has length {len(sequence)}; expected {width}"
            )
        for column, symbol in enumerate(sequence):
            encoded[row, column] = encode_symbol(symbol, nonstandard)
    return encoded, dict(nonstandard)


def mature_alignment(
    text: str,
    target: str = TARGET_SEQUENCE,
    start: int = MATURE_START,
    end: int = MATURE_END,
) -> tuple[np.ndarray, MsaAudit]:
    """Convert fixed-column A2M text to the mature integer matrix and an audit."""
    if len(target) != end or start < 1 or end < start:
        raise ValueError("crop bounds must address the target sequence")
    records = read_a2m(text)
    header, first = records[0]
    if len(first) != len(target):
        raise MsaFormatError(
            f"expected {len(target)} fixed alignment columns, found {len(first)}"
        )
    if target == TARGET_SEQUENCE and header != OFFICIAL_HEADER:
        raise MsaFormatError(
            f"first header is {header!r}; expected {OFFICIAL_HEADER!r}"
        )
    if first.upper() != target:
        raise MsaFormatError("query sequence differs from target")
    encoded, nonstandard = encode_fixed_columns(records)
    mature = encoded[:, start - 1 : end]
    lowercase_positions = tuple(
        index + 1
        for index, symbol in enumerate(first)
        if symbol.islower() and start <= index + 1 <= end
    )
    gap_fraction = np.mean(mature == GAP_CODE, axis=0).astype(np.float64)
    audit = MsaAudit(
        n_sequences=encoded.shape[0],
        n_alignment_columns=encoded.shape[1],
        n_uppercase_columns=sum(symbol.isupper() for symbol in first),
        n_lowercase_columns=sum(symbol.islower() for symbol in first),
        mature_lowercase_positions=lowercase_positions,
        nonstandard_mapped_to_gap=nonstandard,
        mature_gap_fraction=gap_fraction,
    )
    return np.ascontiguousarray(mature), audit
