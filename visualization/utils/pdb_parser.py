"""
PDB parsing utilities for the visualization layer.
Extracts coordinates, secondary structure, and B-factors
in formats ready to serve to the frontend.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipeline.align_structure import parse_pdb, ResidueMap, attach_bfactors
import numpy as np


def pdb_to_residue_list(pdb_string: str) -> list[dict]:
    """
    Returns a list of residue dicts for the frontend:
    [{ "index": 0, "resno": 1, "chain": "A", "aa": "M",
       "x": 12.3, "y": 4.5, "z": 6.7 }, ...]
    """
    rmap = parse_pdb(pdb_string)
    result = []
    for i, (aa, resno, chain, coord) in enumerate(
        zip(rmap.sequence, rmap.residue_ids, rmap.chain_ids, rmap.ca_coords)
    ):
        result.append({
            "index": i,
            "resno": resno,
            "chain": chain,
            "aa": aa,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
        })
    return result


def pdb_with_attention_bfactors(pdb_string: str, scores: list[float]) -> str:
    """
    Returns a modified PDB string with attention scores in the B-factor column.
    Enables native B-factor-based coloring in Mol*/NGL.
    """
    return attach_bfactors(pdb_string, np.array(scores, dtype=np.float32))


AMINO_ACID_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartate",
    "C": "Cysteine", "Q": "Glutamine", "E": "Glutamate", "G": "Glycine",
    "H": "Histidine", "I": "Isoleucine", "L": "Leucine", "K": "Lysine",
    "M": "Methionine", "F": "Phenylalanine", "P": "Proline", "S": "Serine",
    "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
}

AMINO_ACID_PROPERTIES = {
    "A": "nonpolar", "V": "nonpolar", "I": "nonpolar", "L": "nonpolar",
    "M": "nonpolar", "F": "aromatic", "W": "aromatic", "P": "nonpolar",
    "G": "nonpolar", "S": "polar", "T": "polar", "C": "polar",
    "Y": "aromatic", "N": "polar", "Q": "polar", "D": "negative",
    "E": "negative", "K": "positive", "R": "positive", "H": "positive",
}
