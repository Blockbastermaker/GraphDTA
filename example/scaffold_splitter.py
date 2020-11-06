import sys, os
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd


def generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def splitter(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kwargs):
    """Function for doing data splits by chemical scaffold.
        Referred Deepchem for the implementation,  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    """

    print("Using DeepChem Scaffold")
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test,
                                      1.)

    seed = kwargs.get('seed', None)
    #smiles_list = kwargs.get('smiles_list')
    include_chirality = kwargs.get('include_chirality')

    rng = np.random.RandomState(seed)

    scaffolds = {}
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in
                     sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_inds, valid_inds, test_inds = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    return np.array(train_inds), np.array(valid_inds), np.array(test_inds)


if __name__ == "__main__":
    csvfile = sys.argv[1]

    df = pd.read_csv(csvfile, header=None, index_col=None)

    smiles = list(df.values[:, 2])

    train, validate, test = splitter(smiles, seed=1)
    print(train, validate, test)

    names = ['train', 'validate', 'test']
    for i, ndx in enumerate([train, validate, test]):
        dat = df.values[np.array(ndx)]
        dat = pd.DataFrame(dat)
        dat.to_csv(csvfile+"_"+names[i], header=False, index=False)

    print("split molecules done!")
