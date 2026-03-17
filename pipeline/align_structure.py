"""
Maps sequence residue indices to 3D Cα coordinates from a PDB string.

Provides the bridge between:
  - attention[i][j]  (sequence-space)
  - CA_coords[i], CA_coords[j]  (3D-space)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Optional

import numpy as np


@dataclass
class ResidueMap:
    """Alignment between sequence indices and 3D coordinates."""

    sequence: str                    # amino acid sequence (length N)
    residue_ids: list[int]           # PDB residue numbers
    chain_ids: list[str]             # chain identifiers
    ca_coords: np.ndarray            # [N, 3]  Cα positions
    plddt: Optional[np.ndarray]      # [N]  per-residue confidence

    @property
    def n(self) -> int:
        return len(self.sequence)

    def distance_matrix(self) -> np.ndarray:
        """[N, N] Euclidean Cα distance matrix."""
        diff = self.ca_coords[:, None, :] - self.ca_coords[None, :, :]   # [N, N, 3]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def contact_mask(self, threshold_angstrom: float = 8.0) -> np.ndarray:
        """[N, N] boolean — True if residues within threshold Å."""
        return self.distance_matrix() < threshold_angstrom


def parse_pdb(pdb_string: str) -> ResidueMap:
    """
    Extract Cα atoms from a PDB string.
    Uses simple string parsing (no external dependency beyond numpy).
    Falls back to Biopython if available for robustness.
    """
    try:
        return _parse_with_biopython(pdb_string)
    except Exception:
        return _parse_naive(pdb_string)


def _parse_with_biopython(pdb_string: str) -> ResidueMap:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder, three_to_one

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_string))

    _THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "U",
    }

    seq = []
    res_ids = []
    chain_ids = []
    ca_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in _THREE_TO_ONE:
                    continue
                if "CA" not in residue:
                    continue
                seq.append(_THREE_TO_ONE[res_name])
                res_ids.append(residue.get_id()[1])
                chain_ids.append(chain.get_id())
                ca_list.append(residue["CA"].get_vector().get_array())
        break  # first model only

    return ResidueMap(
        sequence="".join(seq),
        residue_ids=res_ids,
        chain_ids=chain_ids,
        ca_coords=np.array(ca_list, dtype=np.float32),
        plddt=None,
    )


def _parse_naive(pdb_string: str) -> ResidueMap:
    """Minimal PDB parser — no external deps."""
    _THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "U",
    }

    seen: dict[tuple, bool] = {}
    seq = []
    res_ids = []
    chain_ids = []
    ca_list = []

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_name = line[17:20].strip()
        chain_id = line[21].strip()
        res_seq = int(line[22:26].strip())

        key = (chain_id, res_seq, res_name)
        if key in seen:
            continue
        seen[key] = True

        aa = _THREE_TO_ONE.get(res_name)
        if aa is None:
            continue

        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        seq.append(aa)
        res_ids.append(res_seq)
        chain_ids.append(chain_id)
        ca_list.append([x, y, z])

    return ResidueMap(
        sequence="".join(seq),
        residue_ids=res_ids,
        chain_ids=chain_ids,
        ca_coords=np.array(ca_list, dtype=np.float32),
        plddt=None,
    )


def attach_bfactors(pdb_string: str, scores: np.ndarray) -> str:
    """
    Replace B-factor column in PDB with attention scores [0..100].
    Enables native B-factor coloring in Mol* / NGL.
    """
    scores_norm = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-8) * 100

    lines = []
    res_map: dict[tuple, float] = {}

    # Build mapping first (parse structure to get index order)
    rmap = _parse_naive(pdb_string)
    for idx, (res_id, chain_id) in enumerate(zip(rmap.residue_ids, rmap.chain_ids)):
        if idx < len(scores_norm):
            res_map[(chain_id, res_id)] = scores_norm[idx]

    for line in pdb_string.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            chain_id = line[21].strip()
            try:
                res_seq = int(line[22:26].strip())
                score = res_map.get((chain_id, res_seq), 0.0)
                line = line[:60] + f"{score:6.2f}" + line[66:]
            except (ValueError, IndexError):
                pass
        lines.append(line)

    return "\n".join(lines)
