import argparse
import sys, os
import pandas as pd
import numpy as np
import json,pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        print("INFO: error! loading smiles error ", smile)
        mol = Chem.MolFromSmiles("CC")

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def seq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    max_seq_len = 1000

    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def smile2graph_dict(smiles_list):
    """
    Convert a list of smiles to a dictionary of smiles-graphs
    """
    compound_iso_smiles = set(smiles_list)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    return smile_graph


def smi2isosmi(smi):
    try:
        new_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("INFO: error! convert smiles failed ", smi)
        new_smi = "C"

    return new_smi


def featurize_dataset(csvfile, output_file="output", dataset_prefix="data", fasta_dir="fastas"):
    # suppose: id, fasta, molid, smiles, pkx
    df = pd.read_csv(csvfile, header=0, index_col=None)
    print("dataset shape: ", df.shape)
    print("dataset header: ,", df.head())

    target_list = df.values[:, 0]
    molid_list = df.values[:, 1]

    train_drugs =np.asarray([smi2isosmi(x) for x in df.values[:, 2]])
    smile2graph_dictionary = smile2graph_dict(train_drugs)
    print("processing total number of compounds: ", train_drugs.shape[0])

    train_protein_ids = df.values[:, 0]
    fasta_sequence_dict = fasta_dict(fasta_dir, set(list(train_protein_ids)))
    train_prots = np.asarray([fasta_sequence_dict[x] for x in train_protein_ids])

    train_Y = df.values[:, -1]
    fasta_encoding = np.asarray([seq_cat(t) for t in train_prots])
    print("processing total number of fasta sequences: ", fasta_encoding.shape[0])

    TestbedDataset(root=dataset_prefix, dataset=output_file, xd=train_drugs,
                   xt=fasta_encoding, y=train_Y, smile_graph=smile2graph_dictionary)

    return target_list, molid_list

def arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default='input.csv', type=str, help='input csv file')
    parser.add_argument("-o", default='output.pt', type=str, help='output filename in pytorch format')
    parser.add_argument("-f", default='fasta_file.fasta', type=str,
                        help='a fasta file containing the target fasta')

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


def fasta_dict(fasta_dir, prot_ids):

    fasta_seq_dict = {}
    for p in prot_ids:
        fn = os.path.join(os.path.join(fasta_dir, p+".fasta"))
        if os.path.exists(fn):
            fasta_seq_dict[p] = get_fasta_seq(fn)
        else:
            print("fasta file not exists: ", fn)
            fasta_seq_dict[p] = "X"

    return fasta_seq_dict


def get_fasta_seq(fasta_file):
    try:
        with open(fasta_file) as lines:
            seq = [x.strip("\n") for x in lines if len(x) and x[0] != ">"]
            seq = "".join(seq)
    except:
        print("fasta file processing error: ", fasta_file)
        seq = "X"

    return seq


if __name__ == "__main__":

    args = arguments()
    if not os.path.exists(args.i):
        print("input file not exists: ", args.i)
        sys.exit(0)

    dirname = os.path.dirname(args.o)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    outname = os.path.basename(args.i)

    featurize_dataset(args.i, dataset_prefix=dirname, output_file=outname)

    print("Featurization completed...")
