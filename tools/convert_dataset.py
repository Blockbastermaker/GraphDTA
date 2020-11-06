import pandas as pd
import sys, os
import numpy as np

def process_csv(csvfile, csvout, kikd="", fasta_dir="fasta"):
    if not os.path.isdir(fasta_dir):
        os.mkdir(fasta_dir)

    df = pd.read_csv(csvfile, header=0, index_col=0, sep="\t")

    features = ['PDB ID(s) of Target Chain', 'ChEMBL ID of Ligand', 'Ligand SMILES']
    sequence = "BindingDB Target Chain  Sequence"
    if kikd == "":
        features.append('pKi_[M]')
    else:
        features.append(kikd)

    df = df[features+[sequence]].drop(axis=0)
    print("dataset shape", df.shape)

    dat = df[features]
    seqs= df[sequence].values
    dat.columns = ['target', 'molid', 'smiles', 'pkx']
    dat['target'] = [ x.split(",")[0] for x in dat['target'].values]

    assert seqs.shape[0] == dat.shape[0]
    for i in range(seqs.shape[0]):
        t = dat['target'].values[i]
        fasta_file = os.path.join(fasta_dir, t+".fasta")
        if not os.path.exists(fasta_file):
            with open(fasta_file, 'w') as tofile:
                tofile.write(">%s \n" % t)
                tofile.write("%s\n" % seqs[i])
            tofile.close()

    dat.to_csv(csvout, header=1, index=0)

    return None


if __name__ == "__main__":

    if len(sys.argv) <= 2:
        print("usage: script.py inp.csv outcsv fasta_dir \"pKi_[M]\"")
        sys.exit(0)

    csvin = sys.argv[1]
    csvout= sys.argv[2]
    fsout = sys.argv[3]
    pkinm = sys.argv[4]

    process_csv(csvin, outcsv, fasta_dir=fsout, kikd=pkinm)

