import collections
import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from CifFile import ReadCif
from tqdm import tqdm

from PDBData import nucleic_letters_3to1_extended, protein_letters_3to1_extended

all_aa_atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O', 'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}
aa2hydrogens = {
    "ALA": (" H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None),
    "ARG": (" H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"),
    "ASN": (" H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None),
    "ASP": (" H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
    "CYS": (" H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None),
    "GLN": (" H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None),
    "GLU": (" H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None),
    "GLY": (" H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),
    "HIS": (" H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None),
    "ILE": (" H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None),
    "LEU": (" H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None),
    "LYS": (" H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "),
    "MET": (" H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None),
    "PHE": (" H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None),
    "PRO": (" HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None),
    "SER": (" H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None),
    "THR": (" H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None),
    "TRP": (" H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None),
    "TYR": (" H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None),
    "VAL": (" H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None),
    "UNK": (" H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None),
    # "  N": (" H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None),# HIS_D
}

all_nucleotide_atom_types = [
    "OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", "N1", "C2", "O2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9", "O4", "O6", "N4", "N2", "C7"
]
nuc2atom = {
    "DA": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C4", "N3", "C2", "N1", "C6", "C5", "N7","C8", "N6"),
    "DC": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "DG": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C4", "N3", "C2", "N1", "C6", "C5", "N7","C8", "N2", "O6"),
    "DT": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7","C6"),
    "DX": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"),  # the O2 above might need to be changed to O2'
    "A":  ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", "N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"),
    "C":  ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "G":  ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"),
    "U":  ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    "RX": ("OP1", "P", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'"),
}
nuc2hydrogen = {
    "DA": ("H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
    "DC": ("H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
    "DG": ("H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
    "DT": ("H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
    "DX": ("H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
    "A":  (" H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
    "C":  (" H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
    "G":  (" H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
    "U":  (" H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
    "RX": (" H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)
}


# All amino acid types to tokens for only the common ones
aatypes = "ARNDCQEGHILKMFPSTWYV"
prot_1letter = set(protein_letters_3to1_extended.values())
assert len(prot_1letter - set(aatypes)) == 0
aa_order = {k: i for i, k in enumerate(aatypes)}
aa_to_token = {_3.replace(" ", ""): aa_order[_1] for _3, _1 in protein_letters_3to1_extended.items()}
aa_to_full_token = {_3.replace(" ", ""): i for i, _3 in enumerate(protein_letters_3to1_extended)}
token_to_aa = restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

# All nucleotides to tokens for only the common ones
nucleotide_types = ["DA", "DC", "DG", "DT", "DX", "A", "C", "G", "U", "RX", "T"]
nu_to_ix = {k: i for i, k in enumerate(nucleotide_types)}
# Not sure if this is the best way...
nucleic_letters_3to1_extended = {**nucleic_letters_3to1_extended, **{k: k for k in nucleotide_types}}
nu_to_token = {_3: nu_to_ix[_1] for _3, _1 in nucleic_letters_3to1_extended.items()}
nu_to_full_token = {k: i for i, k in enumerate(nucleic_letters_3to1_extended)}


def nucleotide_or_aa(code):
    aa = code in aa_to_token
    nu = code in nu_to_token
    if aa and nu:
        raise ValueError(f"code: {code} in both aa_to_token and nu_to_token")
    if aa:
        return "protein"
    elif nu:
        return "dna_rna"
    else:
        return "UNK"


def tokenise_residue(code, restype):
    """Should be a three letter code."""
    if restype == "protein":
        small_ix = aa_to_token.get(code, len(aa_to_token))
        full_ix = aa_to_full_token.get(code, len(aa_to_full_token))
    elif restype == "dna_rna":
        small_ix = nu_to_token.get(code, len(nu_to_token))
        full_ix = nu_to_full_token.get(code, len(nu_to_full_token))
    else:
        raise ValueError(f"restype: {restype} must be in ['AA', 'Nucleotide']")
    return small_ix, full_ix



def get_model(path, download=False):
    if download:
        pdb_string = urllib.request.urlopen(
            f"https://files.rcsb.org/view/{path.upper()}.pdb"
        ).read().decode("ascii")
    else:
        pdb_string = Path(path).read_text()
    with io.StringIO(pdb_string) as pdb_fh:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id='none', file=pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            'Only single model PDBs/mmCIFs are supported. Found'
            f' {len(models)} models.'
        )
    model = models[0]
    return model


class ResidueAtoms:
    res2atoms = collections.defaultdict(lambda: collections.defaultdict(int))
    def __call__(self, model):
        for chain in model:
            for res in chain:
                size = len(self.res2atoms[res.resname])
                for atom in res:
                    self.res2atoms[res.resname][atom.name] += 1
                if len(self.res2atoms[res.resname]) > size:
                    print(f"Found new atom(s) for {res.resname}")


def check_real_pdb_compare_chemical_dict():
    # # from here: https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt (14 Nov 2023)
    # os.system("wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt")
    pdb_codes = [k.split("\t")[0] for k in Path("pdb_entry_type.txt").read_text().split("\n")]
    random.shuffle(pdb_codes)

    print(len(pdb_codes))

    res2atom = ResidueAtoms()

    # it takes 60 mins to do 3000; so would be 70 hrs overall
    for pdb_code in tqdm(pdb_codes[:0]):
        try:
            model = get_model(pdb_code, download=True)
            res2atom(model)
        except Exception as e:
            print(f"{pdb_code}: {e}")

    res_to_atoms = res2atom.res2atoms

    frequency = {k: max(v.values()) for k, v in res_to_atoms.items()}

    max_atoms = {k: len(v.values()) for k, v in res_to_atoms.items()}


    # from here: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
    data = ReadCif("components.cif")  # 23GB RAM required / 30 mins to read

    def extract(info, keys):
        ret = {k: info[k] for k in keys if k in info}
        safe_float = lambda x: float(x) if x != "?" else float("inf")
        def convert(name_format):
            val = info[name_format]
            return [safe_float(v) for v in val] if type(val) == list else safe_float(val)
        if '_chem_comp_atom.model_cartn_x' in info:
            ret["_chem_comp_atom.model_cartn"] = np.stack([convert('_chem_comp_atom.model_cartn_%s' % x) for x in "xyz"])
            ret["_chem_comp_atom.pdbx_model_cartn_ideal"] = np.stack([convert('_chem_comp_atom.pdbx_model_cartn_%s_ideal' % x) for x in "xyz"])
        return ret


    molecule_ids = list(data.keys())
    keys = [
        '_chem_comp.id',
        '_chem_comp.name',
        '_chem_comp_atom.atom_id',
        '_chem_comp_atom.type_symbol',
        '_chem_comp_atom.charge',
        '_chem_comp_bond.atom_id_1',
        '_chem_comp_bond.atom_id_2',
        '_chem_comp_bond.pdbx_stereo_config',
        '_chem_comp_atom.alt_atom_id',
        '_chem_comp_bond.pdbx_ordinal',
    ]
    summary = {
        name: {k: info[k] for k in keys if k in info}
        for name, info in tqdm(data.items(), total=len(data))
    }
    coords_summary = {
        name: extract(info, keys) for name, info in tqdm(data.items(), total=len(data))
    }
    np.save("coords_summary.npy", coords_summary)
    Path("summary.json").write_text(json.dumps(summary))

    # We got: 273 / 273 (when I ran with 300 random pdb files)
    print(f"We got: {len([r for r in res_to_atoms if r.lower() in summary])} / {len(res_to_atoms)}")

    # LR: I'm not sure if atom ids are a function of the molecule... but let's assume they're not
    atoms = set()
    for v in summary.values():
        if '_chem_comp_atom.atom_id' in v:
            atoms = atoms.union(set(v['_chem_comp_atom.atom_id']))
    print(f"{len(atoms)} unique atoms")  # 9338 unique atoms

    stringy = lambda x: '\n'.join([f'{k}:\n\t{",".join(v) if type(v) is list else v}' for k, v in summary[x].items()])
    print(f"rare nucleic acid: {stringy('a38')}\n\n")
    print(f"rare amino acid: {stringy('a30')}\n\n")


coords_summary = np.load("coords_summary.npy", allow_pickle=True).item()


failed = []
element_set = set()
k = '_chem_comp_atom.type_symbol'
for c in coords_summary.values():
    if k in c:
        element_set = element_set.union(set(c[k]))
    else:
        failed.append(c)
# print(f"Failed: {failed}")
all_elements = list(element_set)
element_to_index = {e: i for i, e in enumerate(all_elements)}


def tokenise_element(element_code):
    return element_to_index.get(element_code, len(all_elements))


nucleotide_code_to_index = {k: i for i, k in enumerate(all_nucleotide_atom_types)}
amino_acid_code_to_index = {k: i for i, k in enumerate(all_aa_atom_types)}
atom_index_maps = {"protein": amino_acid_code_to_index, "dna_rna": nucleotide_code_to_index}

index_name_map = list(coords_summary.keys())
name_index_map = {k: i for i, k in enumerate(index_name_map)}


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.
all_chain_types = ["protein", "dna_rna"]
