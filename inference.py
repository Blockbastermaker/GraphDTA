import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import subprocess as sp
from models.ginconv import GINConvNet
from models.ginconv_embed import GINConvNetEmbed
from utils import *
import argparse
from prepare_dataset import *


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="input.csv", help="input csv file.")
    parser.add_argument("-f", default='fasta_file.fasta', type=str,
                        help='a fasta file containing the target fasta')
    parser.add_argument("-d", type=str, default='data/', help="intermediate feature pt file")
    parser.add_argument("-o", type=str, default='predicted.csv', help="output predicted values")
    parser.add_argument("-e", type=str, default='perform_out.csv', help="evaluation output csv file")
    parser.add_argument("-m", type=str, default='pretrained_model.model', help='pretrained model file')
    parser.add_argument("-mi", type=int, default=1, help='1: original GraphDTA; 0: embeded GraphDTA')

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


if __name__ == "__main__":

    args = arguments()

    # dataset preparation
    dirname = os.path.dirname(args.d)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    outname = os.path.basename(args.i)

    if os.path.exists(os.path.join(dirname, "processed/" + outname+".pt")):
        print("find previous generated file", dirname, outname)
    else:
        if args.mi == 1:
            modeling = GINConvNet
            targets, molids = featurize_dataset(args.i, dataset_prefix=dirname,
                                                output_file=outname, fasta_dir=args.f)
        else:
            modeling = GINConvNetEmbed
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smile2embed")
            cmd = "export CUDA_VISIBLE_DEVICES=0 && python %s/prepare_dataset_xde.py -i %s -o %s -f %s" % \
                  (root_dir, args.i, args.d, args.f)
            print("running cmd: ", cmd)
            job = sp.Popen(cmd, shell=True)
            job.communicate()

            #targets, molids = prepare_dataset_xde.featurize_dataset(args.i, dataset_prefix=dirname,
            #                                                        output_file=outname, fasta_dir=args.f)
    print("Featurization completed...")

    cuda_name = "cuda:0"
    print('cuda_name:', "cuda:0")

    TEST_BATCH_SIZE = 512
    pt_file_basename = outname
    pt_file_dirname  = dirname

    if args.mi == 1:
        test_data = TestbedDataset(root=pt_file_dirname, dataset=pt_file_basename)
    else:
        test_data = SmileEmbeddingDataset(root=pt_file_dirname, dataset=pt_file_basename)
        print("loading dataset with SmilesEmbedding", pt_file_basename)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    result = []
    model_st = modeling.__name__
    print('\npredicting for ', pt_file_basename, ' using ', model_st)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)

    model_file_name = args.m
    print("loading model file: ", model_file_name)

    if os.path.exists(model_file_name):
        model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
        G, P = predicting(model, device, test_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        ret = [pt_file_basename, model_st] + [round(e, 3) for e in ret]
        result += [ret]

        assert P.shape[0] == len(molids)
        data_out = pd.DataFrame()
        data_out['target'] = targets
        data_out['molid'] = molids
        data_out['pred_pkx'] = P

        data_out.to_csv(args.o + "_" + model_st)
    else:
        print('model is not available!')

    with open(args.e,'w') as f:
        f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
        for ret in result:
            f.write(','.join(map(str,ret)) + '\n')
