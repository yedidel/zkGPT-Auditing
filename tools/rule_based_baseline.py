"""Rule-based static-analysis baseline for zkml-audit-benchmark.

Scans a ZKML codebase for the canonical patterns associated with the soundness
guards that the benchmark's vulnerability artifacts target. Reports per-pattern
hits and an overall "soundness checklist" score, intended as a pre-LLM lower
bound for any new audit agent submitted against the benchmark.

The intent is NOT to be a competitive auditor: it cannot reason about semantic
omissions (e.g., a witness that is committed but to the wrong vector). It IS
intended to catch the most blatant *syntactic* omissions, so that any LLM-based
auditor reporting headline numbers can be compared against a deterministic,
zero-cost floor.

Usage:
    python rule_based_baseline.py --codebase /path/to/codebase --report out.json

Patterns checked:
    1. Fiat-Shamir transcript binding before each challenge draw
       (presence of `fs_challenge*`, `transcript.absorb*`, `Hash::update*`
        adjacent to challenge generation; flags `random_vec()` / `rand()` calls
        that are not preceded by a transcript absorb)
    2. Commit-before-challenge ordering for lookup arguments
       (a `commit*` / `Pippenger*` call must precede the first random oracle
        query in any function that runs a lookup)
    3. Lookup-argument invocation
       (presence of any of: `Lasso`, `Surge`, `Spice`, `lookup`, `tlookup`,
        `LogUp`, `Plookup`, in any source file that implements a nonlinear
        layer such as Softmax / GeLU / LayerNorm / rmsnorm)
    4. Memcheck companion calls (Spice or equivalent)
       (a `memcheck*` / `multiset*` / `init_hash` etc. call in the same file
        that has any of the lookup invocations from #3)
    5. Per-stage output commitment + cross-stage chain
       (every implementation file for an attention / FFN / norm stage writes
        a `.com` or `output_com` artifact and reads the upstream one)

Each pattern produces a binary signal per (file, function) pair. The script
aggregates to a per-codebase soundness checklist with an overall pass rate.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# ---- Pattern catalogue ------------------------------------------------------


CHALLENGE_DRAW = re.compile(
    r"\b(random_vec|rand_fr|sample_random|draw_challenge|random_challenge|"
    r"hash_to_field|fs_challenge|fs_challenge_vec)\s*\(",
    re.IGNORECASE,
)

TRANSCRIPT_ABSORB = re.compile(
    r"\b(transcript\.absorb|transcript_append|absorb_witness|absorb_into|"
    r"hash\.update|sha256_update|fs_absorb)\s*\(",
    re.IGNORECASE,
)

COMMIT_CALL = re.compile(
    r"\b(commit_int|commit_fr_hyrax|prover_commit|Pedersen::commit|"
    r"Hyrax::commit|commit_table|commit_and_absorb|Pippenger)\s*\(",
    re.IGNORECASE,
)

LOOKUP_NAMES = re.compile(
    r"\b(Lasso|Surge|Spice|lookup|tlookup|LogUp|Plookup|cq|Caulk)\b",
    re.IGNORECASE,
)

NONLINEAR_LAYER = re.compile(
    r"\b(softmax|gelu|relu|layer.?norm|rms.?norm|sigmoid|tanh|exp_table)\b",
    re.IGNORECASE,
)

MEMCHECK = re.compile(
    r"\b(memcheck|multiset|init_hash|read_hash|write_hash|final_hash|"
    r"grand_product)\b",
    re.IGNORECASE,
)

OUTPUT_CHAIN = re.compile(
    r"\b(output_com|input_com|stage_proof|\.com|\.proof|commit_output|"
    r"absorb_input_commitment_if_exists)\b",
    re.IGNORECASE,
)


# ---- Per-codebase analysis --------------------------------------------------


@dataclass
class FileSignals:
    path: str
    has_challenge: bool = False
    has_transcript_absorb: bool = False
    has_commit: bool = False
    has_lookup_name: bool = False
    has_nonlinear_layer: bool = False
    has_memcheck: bool = False
    has_output_chain: bool = False


@dataclass
class CodebaseReport:
    codebase: str
    file_count: int
    files: list[FileSignals] = field(default_factory=list)
    checklist: dict[str, bool] = field(default_factory=dict)
    score: float = 0.0


SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cu", ".cuh", ".rs", ".py", ".sol"}


def scan_file(path: Path) -> FileSignals:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return FileSignals(path=str(path))
    return FileSignals(
        path=str(path),
        has_challenge=bool(CHALLENGE_DRAW.search(text)),
        has_transcript_absorb=bool(TRANSCRIPT_ABSORB.search(text)),
        has_commit=bool(COMMIT_CALL.search(text)),
        has_lookup_name=bool(LOOKUP_NAMES.search(text)),
        has_nonlinear_layer=bool(NONLINEAR_LAYER.search(text)),
        has_memcheck=bool(MEMCHECK.search(text)),
        has_output_chain=bool(OUTPUT_CHAIN.search(text)),
    )


def evaluate_checklist(files: list[FileSignals]) -> dict[str, bool]:
    """Aggregate per-file signals into the five checklist items.

    Each item returns True ("pattern present") or False ("pattern missing").
    A False here is a soundness *suspect*, i.e., the artifact category the
    benchmark targets is at least syntactically un-enforced.
    """
    any_chal = any(f.has_challenge for f in files)
    any_absorb = any(f.has_transcript_absorb for f in files)

    # 1. Fiat-Shamir: every file that draws a challenge must also absorb.
    fs_files = [f for f in files if f.has_challenge]
    fs_ok = all(f.has_transcript_absorb for f in fs_files) if fs_files else True

    # 2. Commit-before-challenge: every file with a lookup name must also commit.
    lookup_files = [f for f in files if f.has_lookup_name]
    commit_ok = all(f.has_commit for f in lookup_files) if lookup_files else False

    # 3. Lookup invocation: any file with a nonlinear layer must have a
    #    lookup-name reference somewhere in the codebase.
    nl_files = [f for f in files if f.has_nonlinear_layer]
    lookup_invoked = any_chal and any(f.has_lookup_name for f in files) if nl_files else False

    # 4. Memcheck companion calls: at least one lookup file must also have memcheck.
    memcheck_ok = any(f.has_lookup_name and f.has_memcheck for f in files)

    # 5. Per-stage output chain: at least one file references stage commitments.
    chain_ok = any(f.has_output_chain for f in files)

    return {
        "fiat_shamir_binding": fs_ok,
        "commit_before_challenge": commit_ok,
        "lookup_invocation": lookup_invoked,
        "memcheck_companion": memcheck_ok,
        "per_stage_output_chain": chain_ok,
    }


def analyse_codebase(root: Path) -> CodebaseReport:
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in SOURCE_SUFFIXES:
            files.append(scan_file(p))
    checklist = evaluate_checklist(files)
    score = sum(1 for v in checklist.values() if v) / max(1, len(checklist))
    return CodebaseReport(
        codebase=str(root),
        file_count=len(files),
        files=files,
        checklist=checklist,
        score=score,
    )


# ---- CLI --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based static-analysis baseline for zkml-audit-benchmark."
    )
    parser.add_argument("--codebase", required=True, type=Path,
                        help="Root directory of the codebase to scan.")
    parser.add_argument("--report", default=None, type=Path,
                        help="Optional path to write a JSON report to.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only the per-pattern checklist + score, no per-file dump.")
    args = parser.parse_args()

    report = analyse_codebase(args.codebase)

    summary = {
        "codebase": report.codebase,
        "file_count": report.file_count,
        "checklist": report.checklist,
        "score": report.score,
    }

    print(json.dumps(summary, indent=2))

    if args.report:
        out = asdict(report)
        args.report.write_text(json.dumps(out, indent=2))
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main()
