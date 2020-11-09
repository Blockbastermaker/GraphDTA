import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv_embed import GINConvNetEmbed
from models.ginconv import GINConvNet
from utils import *
import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", default=['data/processed/train.pt'], nargs="+",
                        type=str, help="train.pt file for training")
    parser.add_argument("-val", default=['data/processed/validate.pt'], nargs="+",
                        type=str, help="validating file for training")
    parser.add_argument("-ts", default=['data/processed/test.pt'], nargs="+", type=str,
                        help="test.pt file for testing")
    parser.add_argument("-o", default='out_model.model', type=str,
                        help="output model for weights")
    parser.add_argument("-mi", type=int, default=0, help="model type: 0 for drug embed, 1 for GINConvNet")
    parser.add_argument("-ne", type=int, default=1000,
                        help="number of epochs for training")
    parser.add_argument("-cuda", type=int, default=-1, help="cuda device id")
    parser.add_argument("-pt", type=str, default='pretrained.model',
                        help="pretrained model to load")
    parser.add_argument("-r", type=str, default='performance.csv',
                        help="test set performance csv")
    parser.add_argument("-l", type=str, default='training.log',
                        help="training log file")
    parser.add_argument("-ct", dest='test_csv', default='test.csv',
                        help="the original dataset containing the target idd information")

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL=20):
    #print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def combine_dataset(files_list, dtype=0):

    datasets = []
    for f in files_list:
        _dataset = load_torch_file(f, dtype)
        if _dataset is not None:
            datasets.append(_dataset)

    final_dataset = None
    if len(datasets) == 1:
        final_dataset = datasets[0]
    elif len(datasets) > 1:
        final_dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        return None

    return final_dataset


def load_torch_file(pt_file, type=0):
    if "processed" in pt_file and os.path.exists(pt_file):
        dirname = os.path.dirname(pt_file).split("/")[0]
        filename = ".".join(os.path.basename(pt_file).split(".")[:-1])

        if type == 0:
            dataset = TestbedDataset(root=dirname, dataset=filename)
            print("loading dataset with TestbedDataset", filename)
        else:
            dataset = SmileEmbeddingDataset(root=dirname, dataset=filename)
            print("loading dataset with SmilesEmbedding", filename)

        return dataset
    else:
        print("the file path format is incorrect, should be data/processed/input.pt")
        return None


def correlation_average(targets, ytrue, ypred):
    targets = targets[1:]
    if targets.ravel().shape[0] == ypred.shape[0]:
        df = pd.DataFrame()
        df['target'] = targets
        df['ytrue'] = ytrue
        df['ypred'] = ypred

        unique_targets = sorted(list(set(list(targets))))
        rp_values = []
        rs_values = []
        for t in unique_targets:
            dat = df[df['target'] == t]
            #print(dat.head())
            try:
                _rp = pearson(dat['ytrue'].values, dat['ypred'].values)
                _rs = spearman(dat['ytrue'].values, dat['ypred'].values)
            except:
                _rp = 0.0
                _rs = 0.0

            rp_values.append(_rp)
            rs_values.append(_rs)

        return np.mean(rp_values), np.mean(rs_values), rp_values, rs_values, unique_targets
    else:
        print("shape unmatch", targets.shape, ypred.shape)
        return 0.0, 0.0, [], [], []


def main():

    args = arguments()

    if args.mi == 0:
        modeling = GINConvNetEmbed #[GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
    else:
        modeling = GINConvNet

    model_st = modeling.__name__
    print("train model with ", model_st)
    pretrained = args.pt

    if args.cuda >= 0:
        cuda_name = "cuda:%d" % args.cuda
        print('cuda_name:', cuda_name)
    else:
        cuda_name = "cpu"

    if os.path.exists(args.test_csv):
        _targets = pd.read_csv(args.test_csv, header=None,
                               index_col=None).values[:, 0]
        print("total %d targets " % len(set(_targets)))
    else:
        _targets = None

    # parameters
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = args.ne
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    train_data = combine_dataset(args.tr, args.mi)
    valid_data = combine_dataset(args.val, args.mi)
    test_data = combine_dataset(args.ts, args.mi)

    if train_data is None or test_data is None:
        print("Error: empty dataset in train or test ...")
    elif valid_data is None:
        print("Warning: empty dataset in validate, split 0.2 data from train set ...")
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data,
                                                               [train_size, valid_size])
    else:
        print("dataset loaded ......")

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    if os.path.exists(pretrained):
        print("using pretrained model: ", pretrained)
        model.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))
        #model = modeling().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_test_mse = 1000
    best_test_ci = 0
    best_epoch = -1
    model_file_name = args.o
    result_file_name = args.r
    log_file_name = args.l

    log_infor = []

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch,
              loss_fn=loss_fn, LOG_INTERVAL=LOG_INTERVAL)

        G, P = predicting(model, device, valid_loader)
        val_mse, val_rmse, val_pr = mse(G, P), rmse(G, P), pearson(G, P)

        G, P = predicting(model, device, test_loader)
        test_mse, test_rmse, test_pr = mse(G, P), rmse(G, P), pearson(G, P)
        if _targets is None:
            rp, rs = 0.0, 0.0
        else:
            rp, rs, _, _, _ = correlation_average(_targets, G, P)

        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch 

            # save best model
            torch.save(model.state_dict(), model_file_name)

        print("\n=>LastBest %4d MSE=%6.3f | Epoch %4d | Val: MSE=%6.3f R=%6.3f | Test: MSE=%6.3f R=%6.3f AR(P)=%.3f AR(S)=%.3f\n" %
              (best_epoch, best_mse, epoch, val_mse, val_pr, test_mse, test_pr, rp, rs))

        log_infor.append([best_epoch, best_mse, epoch, val_rmse, val_mse, val_pr, test_rmse, test_mse, test_pr, rp, rs])

        # save log file
        if epoch % 10 == 0:
            df = pd.DataFrame(log_infor, columns=['best_epoch', 'best_mse', 'epoch', 'v_rmse',
                                                  'v_mse', 'v_r', 't_rmse', 't_mse', 't_r', 'aver_rp', 'aver_rsp'])
            df.to_csv(log_file_name)

        if epoch - best_epoch >= 100:
            print("no improve for 20 epochs, break now")
            break
    G, P = predicting(model, device, test_loader)
    _, _, rpd, rsd, _ts_set = correlation_average(_targets, G, P)

    for i,t in enumerate(_ts_set): 
        print("%s %.3f %.3f" % (t, rpd[i], rsd[i]))

if __name__ == "__main__":
    main()
