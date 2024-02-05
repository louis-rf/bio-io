import os
import random
import unittest
import urllib.request
from pathlib import Path

from tqdm import tqdm

from pdb_io import parse_pdb, to_pdb

testdir = "test_pdbfiles"
recondir = "test_pdbfiles_reconstructed"


def download(n=10):
    os.makedirs(testdir, exist_ok=True)
    pdb_codes = [k.split("\t")[0] for k in Path("pdb_entry_type.txt").read_text().split("\n")]
    random.shuffle(pdb_codes)
    for code in tqdm(pdb_codes[:n]):
        try:
            pdb_string = urllib.request.urlopen(
                f"https://files.rcsb.org/view/{code.upper()}.pdb"
            ).read().decode("ascii")
            Path(os.path.join(testdir, f"{code}.pdb")).write_text(pdb_string)
        except Exception as e:
            print(e)


class CustomMeta(type):
    def __new__(mcs, name, bases, class_dict):
        def create_test_method(pdb_filename):
            os.makedirs(recondir, exist_ok=True)
            def test_method(self):
                pdb_string = Path(pdb_filename).read_text()
                assembly, info = parse_pdb(pdb_string)
                reconstructed_pdb = to_pdb(assembly)
                recon_filename = pdb_filename.replace(testdir, recondir)
                Path(recon_filename).write_text(reconstructed_pdb)
                reparsed_assembly, reparsed_info = parse_pdb(reconstructed_pdb)
                matches = {k: (v == reparsed_info[k]) for k, v in info.items()}
                match_info = {k: v for k, v in info.items() if not matches[k]}
                match_reparsed_info = {k: v for k, v in reparsed_info.items() if not matches[k]}
                assert all(matches.values()), f"Shapes not identical:\n{match_info}\n{match_reparsed_info}"
                # self.assertTrue(all(v == reparsed_info[k] for k, v in info.items()))
            return test_method
        for filename in os.listdir(testdir):
            if not filename.endswith(".pdb"):
                continue
            if os.path.isfile(os.path.join(testdir, filename)):
                test_method = create_test_method(os.path.join(testdir, filename))
                test_name = f"test_{filename.replace('.pdb', '')}"
                class_dict[test_name] = test_method
        return super(CustomMeta, mcs).__new__(mcs, name, bases, class_dict)


class TestFiles(unittest.TestCase, metaclass=CustomMeta):
    pass


if __name__ == '__main__':
    # download(n=100)
    unittest.main()


