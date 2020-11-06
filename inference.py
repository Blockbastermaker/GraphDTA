import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
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

    outname = os.path.basename(args.i)[:-4]
    targets, molids = featurize_dataset(args.i, dataset_prefix=dirname, output_file=outname, fasta_dir=args.f)
    print("Featurization completed...")

    # inference starting from here
    modelings = [GINConvNet, ] #GCNNet]  #[GINConvNet, GATNet, GAT_GCN, GCNNet]

    cuda_name = "cuda:0"
    print('cuda_name:', "cuda:0")
    datasets = ['3fam', 'kiba']

    TEST_BATCH_SIZE = 512
    pt_file_basename = outname
    pt_file_dirname  = dirname

    test_data = TestbedDataset(root=pt_file_dirname, dataset=pt_file_basename)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    result = []
    for modeling in modelings:
        model_st = modeling.__name__
        print('\npredicting for ', pt_file_basename, ' using ', model_st)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        #model_file_name = 'model_' + model_st + '_' + datasets[0] + '.model'
        model_file_name = args.m  #os.path.join(args.m, model_file_name)
        print("loading model file: ", model_file_name)

        if os.path.isfile(model_file_name):
            model.load_state_dict(torch.load(model_file_name))
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            ret = [pt_file_basename, model_st] + [round(e, 3) for e in ret]
            result += [ret]
            #print('dataset,model,rmse,mse,pearson,spearman,ci')
            #print(ret)

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
