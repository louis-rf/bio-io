"""LR: I have noticed one error which remains: if amino acids have insertions like:
52
53
53A
53B
54
"""
import collections
import dataclasses
import io
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from CifFile import ReadCif

import constants_tokenisers as const


@dataclasses.dataclass(frozen=True)
class Chain:
    residue: np.ndarray  # (N,) {residue_types}
    residue_index: np.ndarray  # (N,) {first_residue_index,...,last_residue_index}
    atom_positions: np.ndarray  # (N, num_atom_types, 3) float
    atom_mask: np.ndarray  # .. (N, num_atom_types) {0, 1}
    chain_index: np.ndarray  # .. (N,) {0,..., C}
    b_factors: np.ndarray  # .. (N,) float


@dataclasses.dataclass(frozen=True)
class Water:
    positions: np.ndarray
    b_factors: np.ndarray
    residue_index: np.ndarray  # (num_water,)
    chain_index: np.ndarray  # (num_molecules,)  # annoying thing about PDBs... for some reason molecules belong to different chains....


@dataclasses.dataclass(frozen=True)
class Molecule:
    molecule_id: np.ndarray  # (num_molecules,)
    chain_index: np.ndarray  # (num_molecules,)  # annoying thing about PDBs... for some reason molecules belong to different chains....
    residue_index: np.ndarray  # (num_molecules,)
    atom_mask: np.ndarray  # (num_molecules, max_num_atoms)
    elements: np.ndarray  # (num_molecules, num_atoms(molecule))
    charges: np.ndarray
    positions: np.ndarray
    positions_mask: np.ndarray
    b_factors: np.ndarray
    internal_bonds: np.ndarray  # (num_molecules, num_bonds(molecule), 2)
    internal_bonds_mask: np.ndarray  # (num_molecules, max_num_bonds, 2)
    # bonded_residue_indices: the first index is the residue index that the
    # water molecule is bonded to, the second index is the atom index within that
    # residue


@dataclasses.dataclass(frozen=True)
class Bonds:
    het_residue_indices: np.ndarray  # [num_bonds_to_residues, 2]
    het_water_indices: np.ndarray  # [num_bonds_to_het,]
    het_het_indices: np.ndarray  # [num_bonds_to_het, 2]
    # bonded_residue_indices: the first index is the residue index that the
    # water molecule is bonded to, the second index is the atom index within that
    # residue
    water_residue_indices: np.ndarray  # [num_bonds_to_residues, 2]
    water_water_indices: np.ndarray  # [num_bonds_to_water,]

    residue_residue_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class Assembly:
    """
    N: total number of resolved residues
    C: number of unique chains
    R: number of unique residues <-- Fixed apriori
    E: number of unique elements <-- Fixed apriori
    n: number of single atom molecules
    m: number of small molecules
    b: number of bonds over all small molecules

    index sets:
        residue_types: all biopython residue types {0,...,R} last is unknown.
        element_types: all biopython element types {0,...,E} last is unknown.
        max_charge_difference: the different in the most positively charged ion and least in RCSB.
    """
    protein: Chain
    dna_rna: Chain
    water: Water
    molecule: Molecule
    bonds: Bonds
    #   # Protein / DNA / RNA
    #   residue: np.ndarray  # (N,) {residue_types}
    #   residue_index: np.ndarray  # (N,) {first_residue_index,...,last_residue_index}
    #   restype: np.ndarray  # (N,) nucleic/amino/other
    #   atom_positions: np.ndarray  # (N, num_atom_types, 3) float
    #   atom_mask: np.ndarray  # .. (N, num_atom_types) {0, 1}
    #   chain_index: np.ndarray  # .. (N,) {0,..., C}
    #   b_factors: np.ndarray  # .. (N,) float

    #   # # Ions / single atom molecules
    #   # single_elements: np.ndarray  # (n,) {element_types}
    #   # single_charges: np.ndarray  # (n,) {0,...,max_charge_difference}
    #   # single_positions: np.ndarray  # (n, 3) float

    #   # # Small molecules
    #   # molecule_elements: np.ndarray  # (m,) {0,...,E}
    #   # molecule_charges: np.ndarray  # (m,) {0,...,max_charge_difference}
    #   # molecule_bonds: np.ndarray  # (b,) {0,...,m-1} # double/triple bonds should just be repeated.
    #   # molecule_positions: np.ndarray  # (m, 3) float

    def __post_init__(self):
        if len(np.unique(np.concatenate([self.protein.chain_index, self.dna_rna.chain_index]))) > const.PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {const.PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')


def parse_structure(structure, inter_bonds=None):
    models = list(structure.get_models())
    if len(models) != 1:
        # raise ValueError(
        #     'Only single model PDBs/mmCIFs are supported. Found'
        #     f' {len(models)} models.'
        # )
        warnings.warn(f'Only single model PDBs/mmCIFs are supported. Found {len(models)} models. > Only using first model.')
    model = models[0]

    data = {
        k: {
            "residue_small": [],
            "residue_full": [],
            "atom_positions": [],
            "atom_mask": [],
            "residue_index": [],
            "chain_index": [],
            "b_factors": [],
        }
        for k in const.all_chain_types
    }
    het_molecules = []
    het_single = []
    water_data = []

    def get_majority(chain):
        uniq, counts = np.unique([const.nucleotide_or_aa(res.resname) for res in chain], return_counts=True)
        counts = counts[uniq != "UNK"]
        uniq = uniq[uniq != "UNK"]
        if len(counts):
            return uniq[np.argmax(counts)]
        else:
            return None
    chain_to_type = {chain.id: get_majority(chain) for chain in model}

    # res_atom_indices = {}
    residue_atom_indices = {}
    het_atom_ix_to_molecule = {}
    het_single_indices = {}
    wat_atom_indices = {}
    for chain in model:
        chain_type = chain_to_type[chain.id]
        if chain_type is None:
            continue
        for res in chain:
            # model_id, model_num, chain_id, (hetfield, resseq, insertion_code) = res.full_id
            hetfield, resseq, insertion_code = res.id
            standard_aa_or_nucleic = hetfield == " "
            water = hetfield == "W"
            hetero = hetfield.startswith("H_")
            if standard_aa_or_nucleic:
                residue_small_token, residue_full_token = const.tokenise_residue(res.resname, chain_type)
                atom_index_map = const.atom_index_maps[chain_type]
                pos = np.zeros((len(atom_index_map), 3))
                mask = np.zeros((len(atom_index_map),))
                res_b_factors = np.zeros((len(atom_index_map),))
                for atom in res:
                    if atom.name not in atom_index_map:
                        print(f"Skipping: {atom.name}")
                        continue
                    atom_index = atom.get_serial_number()
                    ai = atom_index_map[atom.name]
                    pos[ai] = atom.coord
                    mask[ai] = 1.
                    res_b_factors[ai] = atom.bfactor

                    # atom_indices.append((atom_index, resseq, ai))
                    # res_atom_indices[atom_index] = (resseq, ai)
                    residue_atom_indices[atom_index] = (chain.id, resseq, ai)
                if np.sum(mask) < 0.5:
                    # If no known atom positions are reported for the residue then skip it.
                    continue
                data[chain_type]["residue_small"].append(residue_small_token)
                data[chain_type]["residue_full"].append(residue_full_token)
                data[chain_type]["atom_positions"].append(pos)
                data[chain_type]["atom_mask"].append(mask)
                data[chain_type]["residue_index"].append(resseq)
                data[chain_type]["chain_index"].append(chain.id)
                data[chain_type]["b_factors"].append(res_b_factors)
            elif water:
                for atom in res:
                    atom_index = atom.get_serial_number()
                    position = atom.coord
                    bfactor = atom.bfactor
                    wat_atom_indices[atom_index] = (chain.id, resseq)
                    water_data.append((resseq, chain.id, position, bfactor))
            elif hetero:
                # Could extract the molecule information/graph from the CIF dictionary
                name = res.resname.lower()
                name_index = const.name_index_map[name]
                if name in const.coords_summary:
                    res_info = const.coords_summary[name]
                    single_atom = '_chem_comp_bond.atom_id_1' not in res_info
                    elements = [const.tokenise_element(e) for e in res_info['_chem_comp_atom.type_symbol']]
                    charges = res_info['_chem_comp_atom.charge']
                    if single_atom:
                        atom = next(iter(res))
                        position = atom.coord
                        bfactor = atom.bfactor
                        het_single.append((name, elements, charges, position, bfactor))
                        het_single_indices[atom.get_serial_number()] = resseq
                    else:
                        # create the graph based on res_info
                        atom_names = {aid: i for i, aid in enumerate(res_info['_chem_comp_atom.atom_id'])}
                        bonds = list(zip(
                            [atom_names[aid] for aid in res_info['_chem_comp_bond.atom_id_1']],
                            [atom_names[aid] for aid in res_info['_chem_comp_bond.atom_id_2']],
                        ))

                        # fill in the known coordinates and bfactors
                        ixs, coords, bfact, atom_ixs = list(zip(*[
                            (atom_names[a.id], a.coord, a.bfactor, a.get_serial_number()) for a in res
                        ]))
                        ixs = list(ixs)
                        positions = np.zeros((len(atom_names), 3), dtype=np.float32)
                        positions_mask = np.zeros(len(atom_names), dtype=bool)
                        bfactors = np.zeros(len(atom_names), dtype=np.float32)
                        positions[ixs] = coords
                        positions_mask[ixs] = True
                        bfactors[ixs] = bfact

                        for i, aix in zip(ixs, atom_ixs):
                            het_atom_ix_to_molecule[aix] = (chain.id, resseq, i)
                        het_molecules.append((name_index, resseq, chain.id, elements, charges, bonds, positions, positions_mask, bfactors))
                else:
                    raise ValueError(f"{res.resname} not in chemical compound dictionary")
            else:
                raise ValueError(f"hetfield unrecognised: {hetfield}")

    # Chain IDs are usually characters so map these to ints.
    chain_ids = data["protein"]["chain_index"] + data["dna_rna"]["chain_index"]
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}

    for ch in const.all_chain_types:
        data[ch]["chain_index"] = np.array([chain_id_mapping[cid] for cid in data[ch]["chain_index"]])
        data[ch]["residue"] = np.array(data[ch]["residue_small"])

    protein = Chain(
        residue=np.array(data["protein"]["residue_small"]),
        residue_index=np.array(data["protein"]["residue_index"]),
        atom_positions=np.array(data["protein"]["atom_positions"]),
        atom_mask=np.array(data["protein"]["atom_mask"]),
        chain_index=np.array(data["protein"]["chain_index"]),
        b_factors=np.array(data["protein"]["b_factors"]),
    )
    dna_rna = Chain(
        residue=np.array(data["dna_rna"]["residue_small"]),
        residue_index=np.array(data["dna_rna"]["residue_index"]),
        atom_positions=np.array(data["dna_rna"]["atom_positions"]),
        atom_mask=np.array(data["dna_rna"]["atom_mask"]),
        chain_index=np.array(data["dna_rna"]["chain_index"]),
        b_factors=np.array(data["dna_rna"]["b_factors"]),
    )

    # # we will have inter-residue bonds: Het--Res, Het-Het and Res-Res.
    # bix = np.array([sum([a1 in res_atom_indices, a2 in res_atom_indices]) for a1, a2 in inter_bonds])
    # het_het = inter_bonds[np.where(bix == 0)]
    # het_res = inter_bonds[np.where(bix == 1)]
    # res_res = inter_bonds[np.where(bix == 2)]

    # Organise water data
    if len(water_data):
        water_resseq, water_chains, water_coords, water_bfactor = list(zip(*water_data))
    else:
        water_resseq, water_chains, water_coords, water_bfactor = [], [], [[]], []

    if type(inter_bonds) != np.ndarray:
        # # We should be in a MMCIF file.
        # bond_type = inter_bonds["_struct_conn.conn_type_id"]

        # chain1 = inter_bonds["_struct_conn.ptnr1_label_asym_id"]
        # seq_num1 = [(i) for i in inter_bonds["_struct_conn.ptnr1_label_seq_id"]]
        # atom_name1 = inter_bonds["_struct_conn.ptnr1_label_atom_id"]

        # chain2 = inter_bonds["_struct_conn.ptnr2_label_asym_id"]
        # seq_num2 = [(i) for i in inter_bonds["_struct_conn.ptnr2_label_seq_id"]]
        # atom_name2 = inter_bonds["_struct_conn.ptnr2_label_atom_id"]

        # wat_wat = ...
        # wat_res = ...

        # bonded_residue_indices = ...
        # bonded_water_indices = ...
        # bonded_het_indices = ...
        raise NotImplementedError(
            "LR: TODO; difficulties, mmcif does not store the bonds atom ids, it stores bond type, "
            "chain_1, chain_2, res_1, res_2, atom_code_1, atom_code_2. This makes it difficult to construct"
            "the same information."
        )
    else:
        # We should be in a PDB file.

        # Trace which atoms the waters are bonded to
        def get_indices(from_type, to_type):  # I think this might need fixing.
            tup = lambda x: x if type(x) == tuple else (x,)
            relevant_bonds = []
            for i, j in inter_bonds:
                if i in from_type and j in to_type:
                    relevant_bonds.append(tup(from_type[i]) + tup(to_type[j]))
                if i in to_type and j in from_type:
                    relevant_bonds.append(tup(from_type[j]) + tup(to_type[i]))
            return np.array(list(set(relevant_bonds)))

        tokenise_chain = lambda x: {k: (chain_id_mapping[v1],) + tuple(v2) for k, (v1, *v2) in x.items()}
        het_atom_ix_to_molecule = tokenise_chain(het_atom_ix_to_molecule)
        residue_atom_indices = tokenise_chain(residue_atom_indices)
        wat_atom_indices = tokenise_chain(wat_atom_indices)

        bonds = Bonds(
            het_residue_indices=get_indices(from_type=het_atom_ix_to_molecule, to_type=residue_atom_indices),  # [..., 4] residue indices are two-fold, so are het molecules, residue, atom
            het_water_indices=get_indices(from_type=het_atom_ix_to_molecule, to_type=wat_atom_indices),  # [..., 3]
            het_het_indices=get_indices(from_type=het_atom_ix_to_molecule, to_type=het_atom_ix_to_molecule),  # [..., 4]
            water_residue_indices=get_indices(from_type=wat_atom_indices, to_type=residue_atom_indices),  # [..., 3] residue indices are two-fold, residue, atom
            water_water_indices=get_indices(from_type=wat_atom_indices, to_type=wat_atom_indices),
            residue_residue_indices=get_indices(from_type=residue_atom_indices, to_type=residue_atom_indices),
        )

    water = Water(
        positions=np.stack(water_coords).astype(np.float32),
        b_factors=np.array(water_bfactor, dtype=np.float32),
        residue_index=np.array(water_resseq, dtype=np.int32),
        chain_index=np.array([chain_id_mapping[ci] for ci in water_chains]),
    )

    # Molecules
    num_molecules = len(het_molecules)
    max_atoms = max((len(elem) for _, _, _ch, elem, *_ in het_molecules), default=0)
    max_bonds = max((len(bnds) for _n, _r, _ch, _e, _c, bnds, _p, _m, _b in het_molecules), default=0)

    atom_mask = np.zeros((num_molecules, max_atoms), dtype=bool)
    resixs = np.zeros((num_molecules,), dtype=np.int32)
    chain_index = np.zeros((num_molecules,), dtype=np.int32)
    elements = np.zeros((num_molecules, max_atoms), dtype=np.int32)
    charges = np.zeros((num_molecules, max_atoms), dtype=np.int32)
    positions = np.zeros((num_molecules, max_atoms, 3), dtype=np.float32)
    positions_mask = np.zeros((num_molecules, max_atoms), dtype=bool)
    b_factors = np.zeros((num_molecules, max_atoms), dtype=np.float32)
    internal_bonds = np.zeros((num_molecules, max_bonds, 2), dtype=np.int32)
    internal_bonds_mask = np.zeros((num_molecules, max_bonds), dtype=bool)

    for i, (name, ri, ci, elem, chrg, bnds, pos, pos_mask, bfct) in enumerate(het_molecules):
        na = len(elem)
        resixs[i] = ri
        chain_index[i] = chain_id_mapping[ci]
        atom_mask[i, :na] = True
        elements[i, :na] = elem
        charges[i, :na] = chrg
        positions[i, :na] = pos
        positions_mask[i, :na] = pos_mask
        b_factors[i, :na] = bfct
        internal_bonds[i, :len(bnds)] = bnds
        internal_bonds_mask[i, :len(bnds)] = True

    molecules = Molecule(
        molecule_id=np.array([n for n, *_ in het_molecules]),
        residue_index=resixs,
        chain_index=chain_index,
        atom_mask=atom_mask,
        elements=elements,
        charges=charges,
        positions=positions,
        positions_mask=positions_mask,
        b_factors=b_factors,
        internal_bonds=internal_bonds,
        internal_bonds_mask=internal_bonds_mask,
    )

    info = {
        "num_amino_acids": protein.residue.shape[0],
        "num_nucleotides": dna_rna.residue.shape[0],
        "num_water": water.b_factors.shape[0],
        "num_het": molecules.molecule_id.shape[0],
        "max_het_atom": molecules.atom_mask.shape[1],
        "water_residue_bonds": bonds.water_residue_indices.shape[0],
        "water_water_bonds": bonds.water_water_indices.shape[0],
        "het_water_bonds": bonds.het_water_indices.shape[0],
        "het_residue_bonds": bonds.het_residue_indices.shape[0],
        "het_het_bonds": bonds.het_het_indices.shape[0],
        "residue_residue_bonds": bonds.residue_residue_indices.shape[0],
    }

    return Assembly(protein=protein, dna_rna=dna_rna, water=water, molecule=molecules, bonds=bonds), info


def parse_pdb(pdb_string):
    def _extract(line):
        nline = line.replace("CONECT", "")
        strings = [nline[5*i:5*(i+1)].replace(" ", "") for i in range(1+(len(nline)//5))]
        return [int(s) for s in strings if len(s)]

    def get_bonds(pdb_data):
        lines = [_extract(line) for line in pdb_data.splitlines() if line.startswith("CONECT")]
        inter_bonds = []
        for src_atom_index, *serial_atom_indices in lines:
            for ix in serial_atom_indices:
                inter_bonds.append((src_atom_index, ix))
        return np.array(inter_bonds)

    with io.StringIO(pdb_string) as fh:
        structure = PDBParser(QUIET=True).get_structure(id='none', file=fh)
    inter_bonds = get_bonds(pdb_string)
    return parse_structure(structure, inter_bonds)


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
            f'{chain_name:>1}{residue_index:>4}')


# based on AlphaFold to_pdb(..)
def to_pdb(assembly: Assembly) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
    prot: The protein to convert to PDB.

    Returns:
    PDB string.
    """
    restypes = list(const.aatypes) + ['X']
    res_1to3 = lambda r: const.token_to_aa.get(restypes[r], 'UNK')
    nuctypes = const.nucleotide_types + ['X']
    nuc_tok2code = lambda r: nuctypes[r].rjust(4)  # retrieve the 1/2 letter code and pad from left
    atom_types = {
        "protein": const.all_aa_atom_types,
        "dna_rna": const.all_nucleotide_atom_types,
    }
    untokenise = {"protein": res_1to3, "dna_rna": nuc_tok2code}

    pdb_lines = []

    chain_index = np.concatenate(
        [assembly.protein.chain_index, assembly.dna_rna.chain_index]
    ).astype(np.int32)

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= const.PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {const.PDB_MAX_CHAINS} chains.')
        chain_ids[i] = const.PDB_CHAIN_IDS[i]

    res_recover_atom_ix = {}

    pdb_lines.append('MODEL     1')
    atom_index = 1

    for chain_type, chain in {"protein": assembly.protein, "dna_rna": assembly.dna_rna}.items():
        # if atom_index > 1 and len(aatype):
        #     # Close the final chain.
        #     pdb_lines.append(_chain_end(atom_index, token_map(aatype[-1]),
        #                                 chain_ids[chain_index[-1]], residue_index[-1]))

        token_map = untokenise[chain_type]

        atom_mask = chain.atom_mask
        atom_positions = chain.atom_positions
        b_factors = chain.b_factors
        aatype = chain.residue
        residue_index = chain.residue_index.astype(np.int32)
        chain_index = chain.chain_index
        if len(chain_index):
            last_chain_index = chain_index[0]

        if np.any(aatype >= len(restypes)):
            raise ValueError('Invalid aatypes.')

        # Add all atom sites.
        for i in range(aatype.shape[0]):
            # Close the previous chain if in a multichain PDB.
            if last_chain_index != chain_index[i]:
                pdb_lines.append(
                    _chain_end(
                        atom_index,
                        token_map(aatype[i - 1]),
                        chain_ids[chain_index[i - 1]],
                        residue_index[i - 1],
                    )
                )
                last_chain_index = chain_index[i]
                atom_index += 1  # Atom index increases at the TER symbol.

            res_name_3 = token_map(aatype[i])#.rjust(3)
            for j, (atom_name, pos, mask, b_factor) in enumerate(zip(
                atom_types[chain_type], atom_positions[i], atom_mask[i], b_factors[i]
            )):
                if mask < 0.5:
                    continue

                record_type = 'ATOM'
                name = atom_name if len(atom_name) == 4 else f' {atom_name}'
                # LR: hack...
                name = f"{name:<4}" if chain_type == "protein" else f"{atom_name:<3}"
                alt_loc = ''
                insertion_code = ''
                occupancy = 1.00
                element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                charge = ''
                res_recover_atom_ix[(chain_index[i], residue_index[i], j)] = atom_index
                atom_line = (f'{record_type:<6}{atom_index:>5} {name}{alt_loc:>1}'
                            f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                            f'{residue_index[i]:>4}{insertion_code:>1}   '
                            f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                            f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                            f'{element:>2}{charge:>2}')
                pdb_lines.append(atom_line)
                atom_index += 1

    # PDB line format
    # cols = [0, 6, 11, 16, 17, 20, 22, 26, 27, 38, 46, 54, 60, 66, 78, 80]

    if len(aatype) == 0:
        aatype = assembly.protein.residue
        token_map = untokenise["protein"]
        chain_index = assembly.protein.chain_index
        residue_index = assembly.protein.residue_index
    if len(aatype):
        # Close the final chain.
        pdb_lines.append(_chain_end(atom_index, token_map(aatype[-1]),
                                    chain_ids[chain_index[-1]], residue_index[-1]))
    ri = residue_index[-1]
    atom_index += 1

    # HETATOM: other
    het_recover_atom_ix = {}
    molecule = assembly.molecule
    for (
        mol_id,
        res_ix,
        ch_ix,
        atom_mask,
        positions,
        positions_mask,
        b_factors,
        internal_bonds,
        internal_bonds_mask,
    ) in zip(
        molecule.molecule_id,
        molecule.residue_index,
        molecule.chain_index,
        molecule.atom_mask,
        molecule.positions,
        molecule.positions_mask,
        molecule.b_factors,
        molecule.internal_bonds,
        molecule.internal_bonds_mask,
    ):
        [positions, positions_mask, b_factors] = [
            a[atom_mask] for a in (positions, positions_mask, b_factors)
        ]
        internal_bonds = internal_bonds[internal_bonds_mask]

        # Recover the molecule info
        name = const.index_name_map[mol_id]
        res_info = const.coords_summary[name]
        atom_names = res_info['_chem_comp_atom.atom_id']
        elements = res_info['_chem_comp_atom.type_symbol']
        charges = res_info['_chem_comp_atom.charge']
        # pi = 0
        for i, (element, charge, pos, pos_mask, b_factor) in enumerate(
            zip(elements, charges, positions, positions_mask, b_factors)
        ):
            if pos_mask < 0.5:
                continue
            record_type = 'HETATM'
            alt_loc = ''
            insertion_code = ''
            chain_id = chain_ids[ch_ix]
            charge = ''  # ?
            # ri += 1
            occupancy = 1.00
            het_recover_atom_ix[(ch_ix, res_ix, i)] = atom_index
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {atom_names[i]:<4}{alt_loc:>1}'
                        f'{name.upper():>3} {chain_id:>1}'
                        f'{res_ix:>4}{insertion_code:>1}   '
                        f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                        f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                        f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            # pi += pos_mask
            atom_index += 1

    # HETATOM: Water
    wat_recover_atom_ix = {}
    water = assembly.water
    for r, c, p, b in zip(water.residue_index, water.chain_index, water.positions, water.b_factors):
        record_type = 'HETATM'
        name = "O"
        alt_loc = ''
        res_name_3 = "HOH"
        chain_id = chain_ids[c]
        ri += 1
        insertion_code = ''
        occupancy = 1.00
        element = "O"
        charge = ''
        wat_recover_atom_ix[(c, r)] = atom_index
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                    f'{res_name_3:>3} {chain_id:>1}'
                    f'{r:>4}{insertion_code:>1}   '
                    f'{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}'
                    f'{occupancy:>6.2f}{b:>6.2f}          '
                    f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1

    # Fill in the bonds
    #   res_recover_atom_ix (chain, res, atom) -> atom_ix
    #   het_recover_atom_ix (chain, res, atom) -> atom_ix
    #   wat_recover_atom_ix (chain, res) -> atom_ix
    # print(assembly.bonds.het_residue_indices)
    # print([(het_recover_atom_ix[(ci, ri, ai)], res_recover_atom_ix[(cj, rj, aj)]) for ci, ri, ai, cj, rj, aj in assembly.bonds.het_residue_indices])
    atom_atom_ixs = collections.defaultdict(list)
    for ci, ri, ai, cj, rj, aj in assembly.bonds.het_het_indices:
        atom_atom_ixs[het_recover_atom_ix[(ci, ri, ai)]].append(het_recover_atom_ix[(cj, rj, aj)])
    for ci, ri, ai, cj, rj, aj in assembly.bonds.het_residue_indices:
        atom_atom_ixs[het_recover_atom_ix[(ci, ri, ai)]].append(res_recover_atom_ix[(cj, rj, aj)])
    for ci, ri, ai, cj, rj in assembly.bonds.het_water_indices:
        atom_atom_ixs[het_recover_atom_ix[(ci, ri, ai)]].append(wat_recover_atom_ix[(cj, rj)])
    for ci, ri, cj, rj in assembly.bonds.water_water_indices:
        atom_atom_ixs[wat_recover_atom_ix[(ci, ri)]].append(wat_recover_atom_ix[(cj, rj)])
    for ci, ri, cj, rj, aj in assembly.bonds.water_residue_indices:
        atom_atom_ixs[wat_recover_atom_ix[(ci, ri)]].append(res_recover_atom_ix[(cj, rj, aj)])
    for ci, ri, ai, cj, rj, aj in assembly.bonds.residue_residue_indices:
        atom_atom_ixs[res_recover_atom_ix[(ci, ri, ai)]].append(res_recover_atom_ix[(cj, rj, aj)])
    for source, bonded in atom_atom_ixs.items():
        pdb_lines.append((f"CONECT{source:>5}" + "".join([f"{b:>5}" for b in bonded])).ljust(80))

    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.
