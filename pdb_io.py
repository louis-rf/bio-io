import collections
import dataclasses
import io
import json
import os
import random
import re
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from CifFile import ReadCif
from tqdm import tqdm

from PDBData import nucleic_letters_3to1_extended, protein_letters_3to1_extended


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

"""In [13]: data["000"].items()
Out[13]: 
[('_chem_comp.id', '000'),
 ('_chem_comp.name', 'methyl hydrogen carbonate'),
 ('_chem_comp.type', 'NON-POLYMER'),
 ('_chem_comp.pdbx_type', 'ATOMP'),
 ('_chem_comp.formula', 'C2 H4 O3'),
 ('_chem_comp.mon_nstd_parent_comp_id', '?'),
 ('_chem_comp.pdbx_synonyms', '?'),
 ('_chem_comp.pdbx_formal_charge', '0'),
 ('_chem_comp.pdbx_initial_date', '2010-04-27'),
 ('_chem_comp.pdbx_modified_date', '2011-06-04'),
 ('_chem_comp.pdbx_ambiguous_flag', 'N'),
 ('_chem_comp.pdbx_release_status', 'REL'),
 ('_chem_comp.pdbx_replaced_by', '?'),
 ('_chem_comp.pdbx_replaces', '?'),
 ('_chem_comp.formula_weight', '76.051'),
 ('_chem_comp.one_letter_code', '?'),
 ('_chem_comp.three_letter_code', '000'),
 ('_chem_comp.pdbx_model_coordinates_details', '?'),
 ('_chem_comp.pdbx_model_coordinates_missing_flag', 'N'),
 ('_chem_comp.pdbx_ideal_coordinates_details', 'Corina'),
 ('_chem_comp.pdbx_ideal_coordinates_missing_flag', 'N'),
 ('_chem_comp.pdbx_model_coordinates_db_code', '3LIN'),
 ('_chem_comp.pdbx_subcomponent_list', '?'),
 ('_chem_comp.pdbx_processing_site', 'RCSB'),
 ('_chem_comp_atom.comp_id',
  ['000', '000', '000', '000', '000', '000', '000', '000', '000']),
 ('_chem_comp_atom.atom_id',
  ['C', 'O', 'OA', 'CB', 'OXT', 'HB', 'HBA', 'HBB', 'HXT']),
 ('_chem_comp_atom.alt_atom_id',
  ['C', 'O', 'OA', 'CB', 'OXT', 'HB', 'HBA', 'HBB', 'HXT']),
 ('_chem_comp_atom.type_symbol',
  ['C', 'O', 'O', 'C', 'O', 'H', 'H', 'H', 'H']),
 ('_chem_comp_atom.charge', ['0', '0', '0', '0', '0', '0', '0', '0', '0']),
 ('_chem_comp_atom.pdbx_align', ['1', '1', '1', '1', '1', '1', '1', '1', '1']),
 ('_chem_comp_atom.pdbx_aromatic_flag',
  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
 ('_chem_comp_atom.pdbx_leaving_atom_flag',
  ['N', 'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'Y']),
 ('_chem_comp_atom.pdbx_stereo_config',
  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
 ('_chem_comp_atom.model_cartn_x',
  ['32.880',
   '32.160',
   '34.147',
   '33.872',
   '32.419',
   '34.788',
   '33.076',
   '33.555',
   '31.625']),
 ('_chem_comp_atom.model_cartn_y',
  ['-0.090',
   '0.180',
   '-0.940',
   '-2.227',
   '0.429',
   '-2.834',
   '-2.800',
   '-1.969',
   '0.931']),
 ('_chem_comp_atom.model_cartn_z',
  ['51.314',
   '50.105',
   '51.249',
   '50.459',
   '52.564',
   '50.416',
   '50.957',
   '49.438',
   '52.425']),
 ('_chem_comp_atom.pdbx_model_cartn_x_ideal',
  ['-0.456',
   '-0.376',
   '0.662',
   '1.929',
   '-1.663',
   '1.996',
   '1.995',
   '2.748',
   '-2.438']),
 ('_chem_comp_atom.pdbx_model_cartn_y_ideal',
  ['0.028',
   '1.240',
   '-0.720',
   '-0.010',
   '-0.566',
   '0.613',
   '0.618',
   '-0.730',
   '0.013']),
 ('_chem_comp_atom.pdbx_model_cartn_z_ideal',
  ['-0.001',
   '0.001',
   '0.001',
   '-0.001',
   '-0.000',
   '-0.892',
   '0.888',
   '0.002',
   '0.002']),
 ('_chem_comp_atom.pdbx_component_atom_id',
  ['C', 'O', 'OA', 'CB', 'OXT', 'HB', 'HBA', 'HBB', 'HXT']),
 ('_chem_comp_atom.pdbx_component_comp_id',
  ['000', '000', '000', '000', '000', '000', '000', '000', '000']),
 ('_chem_comp_atom.pdbx_ordinal',
  ['1', '2', '3', '4', '5', '6', '7', '8', '9']),
 ('_chem_comp_bond.comp_id',
  ['000', '000', '000', '000', '000', '000', '000', '000']),
 ('_chem_comp_bond.atom_id_1',
  ['C', 'O', 'OA', 'CB', 'CB', 'CB', 'CB', 'OXT']),
 ('_chem_comp_bond.atom_id_2',
  ['OXT', 'C', 'C', 'OA', 'HB', 'HBA', 'HBB', 'HXT']),
 ('_chem_comp_bond.value_order',
  ['SING', 'DOUB', 'SING', 'SING', 'SING', 'SING', 'SING', 'SING']),
 ('_chem_comp_bond.pdbx_aromatic_flag',
  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
 ('_chem_comp_bond.pdbx_stereo_config',
  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
 ('_chem_comp_bond.pdbx_ordinal', ['1', '2', '3', '4', '5', '6', '7', '8']),
 ('_pdbx_chem_comp_descriptor.comp_id',
  ['000', '000', '000', '000', '000', '000', '000']),
 ('_pdbx_chem_comp_descriptor.type',
  ['SMILES',
   'SMILES_CANONICAL',
   'SMILES',
   'SMILES_CANONICAL',
   'SMILES',
   'InChI',
   'InChIKey']),
 ('_pdbx_chem_comp_descriptor.program',
  ['ACDLabs',
   'CACTVS',
   'CACTVS',
   'OpenEye OEToolkits',
   'OpenEye OEToolkits',
   'InChI',
   'InChI']),
 ('_pdbx_chem_comp_descriptor.program_version',
  ['12.01', '3.370', '3.370', '1.7.0', '1.7.0', '1.03', '1.03']),
 ('_pdbx_chem_comp_descriptor.descriptor',
  ['O=C(O)OC',
   'COC(O)=O',
   'COC(O)=O',
   'COC(=O)O',
   'COC(=O)O',
   'InChI=1S/C2H4O3/c1-5-2(3)4/h1H3,(H,3,4)',
   'CXHHBNMLPJOKQD-UHFFFAOYSA-N']),
 ('_pdbx_chem_comp_identifier.comp_id', ['000', '000']),
 ('_pdbx_chem_comp_identifier.type', ['SYSTEMATIC NAME', 'SYSTEMATIC NAME']),
 ('_pdbx_chem_comp_identifier.program', ['ACDLabs', 'OpenEye OEToolkits']),
 ('_pdbx_chem_comp_identifier.program_version', ['12.01', '1.7.0']),
 ('_pdbx_chem_comp_identifier.identifier',
  ['methyl hydrogen carbonate', 'methyl hydrogen carbonate']),
 ('_pdbx_chem_comp_audit.comp_id', ['000', '000']),
 ('_pdbx_chem_comp_audit.action_type',
  ['Create component', 'Modify descriptor']),
 ('_pdbx_chem_comp_audit.date', ['2010-04-27', '2011-06-04']),
 ('_pdbx_chem_comp_audit.processing_site', ['RCSB', 'RCSB'])]"""

""" COMBINED nucleotides and amino acids
from PDBData import nucleic_letters_3to1_extended, protein_letters_3to1_extended


nucl_1letter = set(nucleic_letters_3to1_extended.values())
nucl = 'ATGCU'
assert len(nucl_1letter - set(nucl)) == 0
# since ATCG are in prot_1letter and nucl_1letter, switch nucl to integers
nucl2ix = {k: i for i, k in enumerate(nucl)}

prot = 'TRCMPADYNQGLEKFWIVHS'
prot_1letter = set(protein_letters_3to1_extended.values())
assert len(prot_1letter - set(prot)) == 0
prot2ix = {k: i for i, k in enumerate(prot_1letter, start=len(nucl))}

nucl_prot_to_single = {
    **{_3: prot2ix[_1] for _3, _1 in protein_letters_3to1_extended.items()},
    **{_3: nucl2ix[_1] for _3, _1 in nucleic_letters_3to1_extended.items()},
    **{f"D{_1}": nucl2ix[_1] for _1 in "ATGC"},  # DNA
    **{_1: nucl2ix[_1] for _1 in "AGCU"},  # RNA
}  # <--- atm I map the RNA ACG to the same tokens as DNA. TODO model separately.
single_out = len(set(nucl_prot_to_single.values()))
nucl_prot_to_full = {k: i for i, k in enumerate(nucl_prot_to_single)}


def tokenise_residue(code):
    small_ix = nucl_prot_to_single.get(code, single_out)
    full_ix = nucl_prot_to_full.get(code, len(nucl_prot_to_full))
    return small_ix, full_ix
"""


# nucl_1letter = set(nucleic_letters_3to1_extended.values())
# nucl = 'ATGCU'
# assert len(nucl_1letter - set(nucl)) == 0
# # since ATCG are in prot_1letter and nucl_1letter, switch nucl to integers
# nucl2ix = {k: i for i, k in enumerate(nucl)}

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


# nucl_to_single = {
#     **{_3: nucl2ix[_1] for _3, _1 in nucleic_letters_3to1_extended.items()},
#     **{f"D{_1}": nucl2ix[_1] for _1 in "ATGC"},  # DNA
#     **{_1: nucl2ix[_1] for _1 in "AGCU"},  # RNA
# }
# single_out = len(set(nucl_prot_to_single.values()))
# nucl_prot_to_full = {k: i for i, k in enumerate(nucl_prot_to_single)}


# def tokenise_residue(code):
#     """Should be a three letter code."""
#     small_ix = nucl_prot_to_single.get(code, single_out)
#     full_ix = nucl_prot_to_full.get(code, len(nucl_prot_to_full))
#     return small_ix, full_ix


# for chain in model:
#     for res in chain:
#         tok = tokenise_residue(res.resname)
#         if tok is None:
#             # end chain
#             # only look for small molecules, these can be hydrogen bonded.
#             pass
#     print("-" * 80)

"""
# # From RoseTTAFold2
# # full sc atom representation
# aa2long=[
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
#     (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
#     (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
#     (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
#     (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
#     (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
#     (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
#     (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
#     (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
#     (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
#     (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #21 mask
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
#     (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)
#     (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None), #-1 his_d
# ]
# rows = [
#     'ALA',
#     'ARG',
#     'ASN',
#     'ASP',
#     'CYS',
#     'GLN',
#     'GLU',
#     'GLY',
#     'HIS',
#     'ILE',
#     'LEU',
#     'LYS',
#     'MET',
#     'PHE',
#     'PRO',
#     'SER',
#     'THR',
#     'TRP',
#     'TYR',
#     'VAL',
#     'UNK',
#     'MAS',
#     'DA',
#     'DC',
#     'DG',
#     'DT',
#     'DX',
#     'A',
#     'C',
#     'G',
#     'U',
#     'N',  #'RX',
#     # 'HIS_D',
# ]  # match up to my order tokens / 1 letter codes...
# [unk_ix] = [i for i, r in enumerate(rows) if r == 'UNK']
# residue_atom_indices = [{a.replace(" ", ""): i for i, a in enumerate(row) if a != None} for row in aa2long]
"""

# _shared = ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'"]
# nucleotide_to_atom_codes = {
#     "DA": _shared + ["C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
#     "DC": _shared + ["C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
#     "DG": _shared + ["C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
#     "DT": _shared + ["C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"],
#     "A":  _shared + ["O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
#     "C":  _shared + ["O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
#     "G":  _shared + ["O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
#     "U":  _shared + ["O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]
# }

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

# "DX": (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None)
# "RX": (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None)

# nucleotide_atom_codes = _shared + [
#     "N9","C4'","O3'","O2'",  "P", "N2", "O4'", "N6", "C3'", "N1",
#     "N7", "C5", "N3", "C2","O5'", "O4",  "N4", "C8", "C5'", "C4",
#     "O2", "O6","C2'","C5M","C1'", "C6",
# ]
nucleotide_code_to_index = {k: i for i, k in enumerate(all_nucleotide_atom_types)}
amino_acid_code_to_index = {k: i for i, k in enumerate(all_aa_atom_types)}
atom_index_maps = {"protein": amino_acid_code_to_index, "dna_rna": nucleotide_code_to_index}

# a = collections.defaultdict(set)
# for res in model.child_dict["T"]:
#     a[res.resname] = {atom.name for atom in res}.union(a[res.resname])

index_name_map = list(coords_summary.keys())
name_index_map = {k: i for i, k in enumerate(index_name_map)}


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.
all_chain_types = ["protein", "dna_rna"]

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
    if len(np.unique(np.concatenate([self.protein.chain_index, self.dna_rna.chain_index]))) > PDB_MAX_CHAINS:
      raise ValueError(
          f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
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
        for k in all_chain_types
    }
    het_molecules = []
    het_single = []
    water_data = []

    def get_majority(chain):
        uniq, counts = np.unique([nucleotide_or_aa(res.resname) for res in chain], return_counts=True)
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
                residue_small_token, residue_full_token = tokenise_residue(res.resname, chain_type)
                atom_index_map = atom_index_maps[chain_type]
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
                name_index = name_index_map[name]
                if name in coords_summary:
                    res_info = coords_summary[name]
                    single_atom = '_chem_comp_bond.atom_id_1' not in res_info
                    elements = [tokenise_element(e) for e in res_info['_chem_comp_atom.type_symbol']]
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

    for ch in all_chain_types:
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
            # def flatten(i1, i2):
            #     _from = from_type[i1]
            #     _to = to_type[i2]
            #     out = _from if type(_from) == tuple else (_from,)
            #     return 
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
    # print(info)

    return Assembly(protein=protein, dna_rna=dna_rna, water=water, molecule=molecules, bonds=bonds), info


def parse_pdb(pdb_string):
    # def get_bonds(pdb_data):
    #     pattern = r"^CONECT\s+(\d+)\s+(\d+)?\s*(\d+)?\s*(\d+)?\s*(\d+)?"
    #     matches = re.findall(pattern, pdb_data, re.MULTILINE)
    #     inter_bonds = []
    #     for match in matches:
    #         src_atom_index, *serial_atom_indices = [int(index) for index in match if index]
    #         for ix in serial_atom_indices:
    #             inter_bonds.append((src_atom_index, ix))
    #     return np.array(inter_bonds)
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


# def parse_mmcif(cif_string):
#     def generate_atom_indices(structure):
#         atom_index = 1  # Start from 1 as in PDB files
#         atom_mapping = {}

#         for model in structure:
#             for chain in model:
#                 for residue in chain:
#                     for atom in residue:
#                         # key = (chain.id, residue.id[1], atom.id)  # (Chain ID, Residue sequence number, Atom name)
#                         key = (residue.id[1], atom.id)
#                         atom_mapping[key] = atom_index
#                         atom_index += 1

#         return atom_mapping


#     with io.StringIO(cif_string) as fh:
#         structure = MMCIFParser(QUIET=True).get_structure(structure_id='none', filename=fh)
#     with io.StringIO(cif_string) as fh:
#         mmcif_dict = MMCIF2Dict(fh)
#     atom_indices = generate_atom_indices(structure)
#     for i, conn_type in enumerate(mmcif_dict["_struct_conn.conn_type_id"]):
#         if (
#             mmcif_dict["_struct_conn.ptnr1_label_seq_id"][i] == "." or
#             mmcif_dict["_struct_conn.ptnr2_label_seq_id"][i] == "."
#         ):
#             continue
#         chain1 = mmcif_dict["_struct_conn.ptnr1_label_asym_id"][i]
#         print(mmcif_dict["_struct_conn.ptnr1_label_seq_id"][i])
#         seq_num1 = int(mmcif_dict["_struct_conn.ptnr1_label_seq_id"][i])
#         atom_name1 = mmcif_dict["_struct_conn.ptnr1_label_atom_id"][i]

#         chain2 = mmcif_dict["_struct_conn.ptnr2_label_asym_id"][i]
#         seq_num2 = int(mmcif_dict["_struct_conn.ptnr2_label_seq_id"][i])
#         atom_name2 = mmcif_dict["_struct_conn.ptnr2_label_atom_id"][i]

#         # index1 = atom_indices.get((chain1, seq_num1, atom_name1))
#         # index2 = atom_indices.get((chain2, seq_num2, atom_name2))
#         index1 = atom_indices.get((seq_num1, atom_name1))
#         index2 = atom_indices.get((seq_num2, atom_name2))
#         print(f"{conn_type}: {(chain1, seq_num1, atom_name1)}  !  {index1} - {index2}")
#     return
#     mmcif_dict = None
#     return parse_structure(structure, mmcif_dict)
# _ATOM_FORMAT_STRING = (
#     "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%s%6.2f      %4s%2s%2s\n"
# )
# _PQR_ATOM_FORMAT_STRING = (
#     "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f %7s  %6s      %2s\n"
# )

# _TER_FORMAT_STRING = (
#     "TER   %5i      %3s %c%4i%c                                                      \n"
# )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


# AlphaFold to_pdb(..)
def to_pdb(assembly: Assembly) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
    prot: The protein to convert to PDB.

    Returns:
    PDB string.
    """
    restypes = list(aatypes) + ['X']
    res_1to3 = lambda r: token_to_aa.get(restypes[r], 'UNK')
    nuctypes = nucleotide_types + ['X']
    nuc_tok2code = lambda r: nuctypes[r].rjust(4)  # retrieve the 1/2 letter code and pad from left
    atom_types = {
        "protein": all_aa_atom_types, "dna_rna": all_nucleotide_atom_types,
    }
    untokenise = {"protein": res_1to3, "dna_rna": nuc_tok2code}

    pdb_lines = []

    chain_index = np.concatenate(
        [assembly.protein.chain_index, assembly.dna_rna.chain_index]
    ).astype(np.int32)

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

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
                # segid = ''
                res_recover_atom_ix[(chain_index[i], residue_index[i], j)] = atom_index
                # args = (
                #     record_type,
                #     atom_index,
                #     name,
                #     alt_loc,
                #     res_name_3,
                #     chain_ids[chain_index[i]],
                #     residue_index[i],
                #     insertion_code,
                #     pos[0],
                #     pos[1],
                #     pos[2],
                #     occupancy,
                #     b_factor,
                #     segid,
                #     element,
                #     charge,
                # )
                # atom_line = _ATOM_FORMAT_STRING % args
                # PDB is a columnar format, every space matters here!
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
        name = index_name_map[mol_id]
        res_info = coords_summary[name]
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
    # import pdb;pdb.set_trace()
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



# # pdb_filename = "pdbs/1H4S.pdb"
# pdb_filename = "test_pdbfiles/6cuz.pdb"
# pdb_string = Path(pdb_filename).read_text()
# assembly, info = parse_pdb(pdb_string)
# # assembly2, info2 = parse_mmcif(Path("pdbs/1H4S.cif").read_text())


# # print(info)
# # for k, v in assembly.__dict__.items():
# #     print(k)
# #     for kk, vv in v.__dict__.items():
# #         print(f"\t{kk}: {vv.shape}")


# reconstructed_pdb = to_pdb(assembly)
# # recon_filename = pdb_filename.replace(".pdb", "_reconstructed.pdb")
# # Path(recon_filename).write_text(reconstructed_pdb)

# # os.system(f"git diff {pdb_filename} {recon_filename} > reconstruction_error.diff")


# reparsed_assembly, reparsed_info = parse_pdb(reconstructed_pdb)



# Note: all working apart from the atom indexing system for bonds! PDB can work if we do atom re-indexing



"""Inter-type bonds:

water -> water bonds
water -> residue atom bonds
het -> het bonds
het -> residue atom bonds
het -> water bonds
"""





"""
When you see HOH in a PDB file, it represents a water molecule. Typically, water molecules are not covalently bonded to proteins. However, they can form hydrogen bonds or other non-covalent interactions with the protein, which can be crucial for the protein's structure and function.

The CONECT records that include water molecules might not necessarily represent covalent bonds. They could represent:

Hydrogen Bonds: Water can form hydrogen bonds with the protein, especially with polar or charged amino acid side chains.
Metal Coordination: Some proteins have metal ions (like Zn, Mg, Ca, etc.) that can be coordinated (bonded) with water molecules.
Crystallographic Artifacts: Sometimes, in the process of determining a protein's structure using X-ray crystallography, certain interactions can be observed that might not be present in the protein's natural environment. This could include water molecules being in close proximity to certain atoms in the protein.
Errors or Misinterpretations: It's also possible that some CONECT records might not represent actual biological interactions but could be errors or misinterpretations.
"""




# """Naive datastructure:
# num_residue_types = int every residue type in biopython (includes nucleotides, water, ions,...)
# num_atom_types = unique atom types, the same element in a different residue can be different

# residue: [index of residue type,...] (N,)
# residue_index: [index in sequence,...] (N,)
# atom_positions: .. (N, num_atom_types, 3)
# atom_mask: .. (N, num_atom_types)
# chain_index: .. (N,)
# b_factors: .. (N,)

# The above datastructure can be modified to have:

# num_atom_types as a function of the residue.
# Then we only need to have max(num_atom_types(r) for r in residues)
# along the second axes of atom_*

# If there are small molecules these could be encoded as graphs.

# Another variant could use gromacs / MDAnalysis / some other tool to place the bonds and hydrogens.



# DNA/RNA can be done in the modifier Protein object above.
# All ions could be: ("Element", "Charge", "Position")
# Lipids: (Not sure is these are in PDB files...? maybe in MD files..)
# Small molecules: {"Elements": (n,), "Charges": (n,), "Bonds": (m, 2), "Positions": (n, 3)}

# """
# import numpy as np
# import dataclasses


# class AssemblyMetaData:  # NOTE: DON'T DO META DATA ON THIS! DO IT ON THE FINAL FEATURES...
#     euclidean_dims = 3
#     num_residues = "N"
#     num_chains = "C"
#     num_single_atom_molecules = "n"
#     num_small_molecules = "m"
#     num_small_molecule_bonds = "b"
#     # # Protein / DNA / RNA
#     # residue
#     # residue_index
#     # atom_positions
#     # atom_mask
#     # chain_index
#     # b_factors
#     # ...


# class AssemblyFeaturesMetaData:
#     """
#     """



# def featurise(assembly: Assembly):
#     # construct spatial, radial, buffered cutoffs for interactions of small molecules, ion, residues


#     return
