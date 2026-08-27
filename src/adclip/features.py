"""
Featurization: A-domain --> one-hot encoding and substrate SMILES --> Morgan fingerprint / RDKit descriptors

"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from . import config

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
UNKNOWN_RESIDUE_IDX = len(AA_ALPHABET)  # index 20 of ONEHOT_DIM=21 — "unknown"/unresolved bin

N_DESCRIPTORS = 12


def _residue_onehot(aa: str) -> np.ndarray:
    v = np.zeros(config.ONEHOT_DIM, dtype=np.float32)
    if aa is None:
        v[UNKNOWN_RESIDUE_IDX] = 1.0
        return v
    v[AA_TO_IDX.get(aa.upper(), UNKNOWN_RESIDUE_IDX)] = 1.0
    return v


def build_adomain_onehot(sequence: str, code_idx: list) -> np.ndarray:

    vecs = []
    for idx in code_idx:
        if idx is None or idx < 0 or idx >= len(sequence):
            vecs.append(_residue_onehot(None))
        else:
            vecs.append(_residue_onehot(sequence[idx]))
    return np.stack(vecs).astype(np.float32)


def code_idx_to_residues(sequence: str, code_idx: list) -> str:

    chars = []
    for idx in code_idx:
        if idx is None or idx < 0 or idx >= len(sequence):
            chars.append("-")
        else:
            chars.append(sequence[idx])
    return "".join(chars)


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def compute_morgan_fingerprint(smiles: str, radius: int = 4, n_bits: int = 1024,
                                include_chirality: bool = False) -> np.ndarray:
   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=include_chirality)
    fp = gen.GetFingerprint(mol)
    return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)


def compute_descriptors(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeterocycles(mol),
        Descriptors.NHOHCount(mol),
    ]


def normalize_descriptors(raw: list, desc_mean: list, desc_std: list) -> np.ndarray:
    return (np.array(raw, dtype=np.float64) - np.array(desc_mean, dtype=np.float64)) / np.array(
        desc_std, dtype=np.float64
    )
