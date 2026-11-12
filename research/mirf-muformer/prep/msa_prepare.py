"""Read an A2M alignment and cut the mature 263 match columns.

Column type follows HH-suite and is decided by the first sequence. Uppercase
amino acids and '-' are match states. Lowercase amino acids and '.' are
insertion states and the whole column is removed. A '.' or lowercase letter
inside a match column stops the conversion; it is not treated as a gap.
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
MATURE_START = 24
MATURE_END = 286


class MsaFormatError(ValueError):
    """The alignment cannot be converted without guessing a symbol."""


@dataclass(frozen=True)
class MsaAudit:
    n_sequences: int
    raw_columns: int
    n_match_columns: int
    n_insert_columns: int
    nonstandard_mapped_to_gap: dict[str, int]
    mature_gap_fraction: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "n_sequences": self.n_sequences,
            "raw_columns": self.raw_columns,
            "n_match_columns": self.n_match_columns,
            "n_insert_columns": self.n_insert_columns,
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


def match_column_index(first_sequence: str) -> np.ndarray:
    """Return indices of match columns. Insertion columns are omitted."""
    chosen: list[int] = []
    for index, symbol in enumerate(first_sequence):
        if symbol in AMINO_ACIDS or symbol == "-":
            chosen.append(index)
        elif symbol in AMINO_ACIDS.lower() or symbol == ".":
            continue
        else:
            raise MsaFormatError(
                f"first sequence column {index + 1} has unclassified symbol {symbol!r}"
            )
    return np.asarray(chosen, dtype=np.int64)


def encode_match_columns(
    records: list[tuple[str, str]],
    columns: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode match columns. '-' is gap. '.' or lowercase stops the conversion."""
    encoded = np.empty((len(records), len(columns)), dtype=np.int8)
    nonstandard: Counter[str] = Counter()
    for row, (_, sequence) in enumerate(records):
        for column, source in enumerate(columns):
            symbol = sequence[int(source)]
            if symbol == "-":
                encoded[row, column] = GAP_CODE
            elif symbol in CODE:
                encoded[row, column] = CODE[symbol]
            elif symbol == "." or symbol in AMINO_ACIDS.lower():
                raise MsaFormatError(
                    "match column "
                    f"{column + 1} contains insertion symbol {symbol!r} in row {row + 1}"
                )
            else:
                nonstandard[symbol] += 1
                encoded[row, column] = GAP_CODE
    return encoded, dict(nonstandard)


def mature_alignment(
    text: str,
    target: str = TARGET_SEQUENCE,
    start: int = MATURE_START,
    end: int = MATURE_END,
) -> tuple[np.ndarray, MsaAudit]:
    """Convert A2M text to the mature integer matrix and an audit."""
    if len(target) != end or start < 1 or end < start:
        raise ValueError("crop bounds must address the target sequence")
    records = read_a2m(text)
    columns = match_column_index(records[0][1])
    encoded, nonstandard = encode_match_columns(records, columns)
    if encoded.shape[1] != len(target):
        raise MsaFormatError(
            f"found {encoded.shape[1]} match columns; the target has {len(target)}"
        )
    first = "".join(
        "-" if code == GAP_CODE else AMINO_ACIDS[code - 1]
        for code in encoded[0]
    )
    if first != target:
        raise MsaFormatError("first sequence match states differ from the target")
    mature = encoded[:, start - 1 : end]
    gap_fraction = np.mean(mature == GAP_CODE, axis=0).astype(np.float64)
    audit = MsaAudit(
        n_sequences=encoded.shape[0],
        raw_columns=len(records[0][1]),
        n_match_columns=encoded.shape[1],
        n_insert_columns=len(records[0][1]) - encoded.shape[1],
        nonstandard_mapped_to_gap=nonstandard,
        mature_gap_fraction=gap_fraction,
    )
    return np.ascontiguousarray(mature), audit
