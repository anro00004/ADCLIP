"""
Aligns unaligned A-domain sequences onto the 1AMU reference frame to
recover `code_idx` (the 16 canonical anchor-position indices into each raw
sequence).

"""
import shutil
import subprocess
import tempfile
from pathlib import Path
from . import config

MUSCLE_VERSION_NOTE = "Pinned for reproducibility: MUSCLE 5.3 (see docs/ENVIRONMENT.md)."


class AlignmentError(RuntimeError):
    pass


def _resolve_muscle_binary() -> str:
    binary = shutil.which("muscle")
    if binary is None:
        raise AlignmentError(
            "MUSCLE not found on PATH. This toolkit needs the `muscle` binary "
            "(https://github.com/rcedgar/muscle) for alignment — it is not a "
            "pip package. Install it (e.g. `conda install -c conda-forge muscle`) "
            "and try again. See docs/ENVIRONMENT.md."
        )
    return binary


def read_fasta(path) -> dict:
    records = {}
    current_id = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if current_id is not None:
            records[current_id] = "".join(chunks)
    return records


def write_fasta(records: dict, path):
    with open(path, "w") as f:
        for rec_id, seq in records.items():
            f.write(f">{rec_id}\n{seq}\n")


def run_muscle(input_fasta, output_fasta, threads=16, verbose=True):
    binary = _resolve_muscle_binary()
    cmd = [binary, "-super5", str(input_fasta), "-output", str(output_fasta), "-threads", str(threads)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        if verbose:
            print(f"  [muscle] {line.rstrip()}")
        tail.append(line)
        tail = tail[-200:]
    proc.wait()
    if proc.returncode != 0:
        raise AlignmentError(
            f"muscle failed (exit {proc.returncode}). Command: {' '.join(cmd)}\n"
            f"output (tail):\n{''.join(tail)[-4000:]}"
        )
    return proc


def _unaligned_to_aligned_pos_1AMU(aligned_seq: str, unaligned_pos_1idx: int):
    unaligned_count = 0
    for aligned_idx, aa in enumerate(aligned_seq):
        if aa != "-":
            unaligned_count += 1
            if unaligned_count == unaligned_pos_1idx:
                return aligned_idx + 1
    return None


def get_code_idx(aligned_seq: str, aligned_columns_1idx: list):
    code_idx = []
    for aligned_pos_1idx in aligned_columns_1idx:
        if aligned_pos_1idx is None:
            code_idx.append(None)
            continue
        aligned_idx_0 = aligned_pos_1idx - 1
        if aligned_idx_0 >= len(aligned_seq):
            code_idx.append(None)
            continue
        aa = aligned_seq[aligned_idx_0]
        if aa == "-":
            code_idx.append(None)
            continue
        unaligned_idx = sum(1 for c in aligned_seq[:aligned_idx_0] if c != "-")
        code_idx.append(unaligned_idx)
    return code_idx


def _load_pool_records(pool: str) -> dict:
    if pool == "training":
        return read_fasta(config.DEFAULT_CORPUS_FASTA)
    else:
        pool_path = Path(pool)
    if not pool_path.exists():
        raise AlignmentError(f"pool='{pool}' is neither \"training\" nor an existing FASTA path.")
    return read_fasta(pool_path)



def align_new_sequences(new_sequences: dict, pool: str = "training", threads: int = 4,
                         verbose: bool = True) -> dict:
    """new_sequences: {id -> raw unaligned A-domain sequence}.
    Returns {id -> {"code_idx": [16 ints or None], "unresolved_positions": [subset of config.POSITIONS]}}.
    """
    ref_records = read_fasta(config.REFERENCE_1AMU_FASTA)
    if len(ref_records) != 1:
        raise AlignmentError(f"Expected exactly one record in {config.REFERENCE_1AMU_FASTA}, "
                              f"found {len(ref_records)}.")
    (ref_id, ref_seq), = ref_records.items()

    pool_records = _load_pool_records(pool)

    combined = {ref_id: ref_seq, **pool_records, **new_sequences}
    if len(combined) != 1 + len(pool_records) + len(new_sequences):
        raise AlignmentError("Duplicate sequence IDs across the reference/pool/new-sequence sets — "
                              "every id must be unique.")

    with tempfile.TemporaryDirectory(prefix="adclip_align_") as tmp:
        tmp = Path(tmp)
        in_fasta = tmp / "combined.fasta"
        out_fasta = tmp / "aligned.fasta"
        write_fasta(combined, in_fasta)

        if verbose:
            print(f"  Aligning {len(new_sequences)} new sequence(s) against 1AMU "
                  f"+ pool='{pool}' ({len(pool_records)} sequences) with MUSCLE...")
        run_muscle(in_fasta, out_fasta, threads=threads, verbose=verbose)

        aligned = read_fasta(out_fasta)

    if ref_id not in aligned:
        raise AlignmentError(f"Reference '{ref_id}' missing from MUSCLE output — alignment failed.")
    ref_aligned = aligned[ref_id]

    fresh_cols = {pos: _unaligned_to_aligned_pos_1AMU(ref_aligned, pos) for pos in config.POSITIONS}
    aligned_columns = [fresh_cols[pos] for pos in config.POSITIONS]

    results = {}
    for seq_id in new_sequences:
        if seq_id not in aligned:
            raise AlignmentError(f"New sequence '{seq_id}' missing from MUSCLE output.")
        code_idx = get_code_idx(aligned[seq_id], aligned_columns)
        unresolved = [pos for pos, code in zip(config.POSITIONS, code_idx) if code is None]
        if verbose and unresolved:
            print(f"  [warn] {seq_id}: unresolved anchor position(s) {unresolved} — "
                  f"substituting the model's 'unknown residue' bin for those.")
        results[seq_id] = {"code_idx": code_idx, "unresolved_positions": unresolved}

    return results
