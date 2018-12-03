from __future__ import print_function
import _pickle as pickle
import numpy as np
import io, os
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/home/zzhang12/work/scripts_summary')
import utils_model



################################ load data ################################
class VcmDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(pickle_file, 'rb') as pf:
            meta = pickle.load(pf)
            self.data = np.array(meta['feats']).reshape(-1, 88).astype(np.float32)
            self.labels = meta['labels']
            print(self.data.shape, np.array(self.labels).shape)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.labels[idx]

        # if self.transform:
        #     feature = self.transform(feature)

        return torch.from_numpy(feature), torch.tensor(label)




################################ design model ################################
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(88, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



class Net1L(nn.Module):
    def __init__(self):
        super(Net1L, self).__init__()
        self.fc1 = nn.Linear(88, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




################################ train model ################################
def train(args, model, device, train_loader, optimizer, epoch, model1l, optimizer1l, train_loader1l): #, optimizer1_1l):
    model.train()
    model1l.train()
    # train the model1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward() #retain_graph=True)
        optimizer.step()

    # train the model1l
    for batch_idx, (data1l, target1l) in enumerate(train_loader1l):
        data1l, target1l = data1l.to(device), target1l.to(device)
        optimizer1l.zero_grad()
        # output = model(data1l)  # torch.cat((data, output), 1))
        output1l = model1l(data1l) #torch.cat((data1l, output), 1))
        loss1l = F.nll_loss(output1l, target1l.long())
        loss1l.backward()
        optimizer1l.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, model1l, test_loader1l):
    model.eval()
    model1l.eval()
    test_loss, correct = 0, 0
    targets, predictions = [], []
    test_loss1l, correct1l = 0, 0
    targets1l, predictions1l = [], []
    with torch.no_grad():
        # from model1
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            targets.extend(np.array(target))
            predictions.extend(np.array(pred).reshape(-1))
            correct += pred.eq(target.view_as(pred)).sum().item()

        results = utils_model.cal_performance(targets, predictions, uar=True, war=True)
        print('LING: uar: {}\t war: {}'.format(results['uar'], results['war']))

        # from model1l
        for data1l, target1l in test_loader1l:
            data1l, target1l = data1l.to(device), target1l.to(device)
            # output = model(data1l)
            output1l = model1l(data1l) #torch.cat((data1l, output), 1))
            test_loss1l += F.nll_loss(output1l, target1l, reduction='sum').item()  # sum up batch loss
            pred1l = output1l.max(1, keepdim=True)[1]  # get the index of the max log-probability
            targets1l.extend(np.array(target1l))
            predictions1l.extend(np.array(pred1l).reshape(-1))
            correct1l += pred1l.eq(target1l.view_as(pred1l)).sum().item()

        results1l = utils_model.cal_performance(targets1l, predictions1l, uar=True, war=True)
        print('SYLL: uar: {}\t war: {}'.format(results1l['uar'], results1l['war']))

        return results, results1l



def read_dataset(dataset):
    with open(dataset, 'rb') as infile:
        data = pickle.load(infile, encoding='iso-8859-1')
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    corpus = 'ling' #''ling'
    train_set = '/data/work2/aclew/threeCorporaHtk/' + corpus + '/feat/' + corpus + '.eGeMAPS.func_utt.train.us.pickle'
    devel_set = '/data/work2/aclew/threeCorporaHtk/' + corpus + '/feat/' + corpus + '.eGeMAPS.func_utt.devel.pickle'
    test_set = '/data/work2/aclew/threeCorporaHtk/' + corpus + '/feat/' + corpus + '.eGeMAPS.func_utt.test.pickle'

    train_loader = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=train_set, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    devel_loader = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=devel_set, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=test_set, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    corpus1l = 'syll' #''ling'
    train_set1l = '/data/work2/aclew/threeCorporaHtk/' + corpus1l + '/feat/' + corpus1l + '.eGeMAPS.func_utt.train.us.pickle'
    devel_set1l = '/data/work2/aclew/threeCorporaHtk/' + corpus1l + '/feat/' + corpus1l + '.eGeMAPS.func_utt.devel.pickle'
    test_set1l = '/data/work2/aclew/threeCorporaHtk/' + corpus1l + '/feat/' + corpus1l + '.eGeMAPS.func_utt.test.pickle'

    train_loader1l = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=train_set1l, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    devel_loader1l = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=devel_set1l, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader1l = torch.utils.data.DataLoader(
        VcmDataset(pickle_file=test_set1l, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model1 = Net1().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)

    model1l = Net1L().to(device)
    optimizer1l = optim.SGD(model1l.parameters(), lr=args.lr, momentum=args.momentum)

    # optimizer1_1l = optim.SGD(list(model1.parameters()) + list(model1l.parameters()), lr=args.lr, momentum=args.momentum)

    ### save the best model based on the development set (metric: UAR)
    best_results_devel, best_results_test = {'uar': 0.0, 'war': 0.0}, {'uar': 0.0, 'war': 0.0}
    best_results1l_devel, best_results1l_test = {'uar': 0.0, 'war': 0.0}, {'uar': 0.0, 'war': 0.0}
    for epoch in range(1, args.epochs + 1):
        train(args, model1, device, train_loader, optimizer1, epoch, model1l, optimizer1l, train_loader1l) #, optimizer1_1l)
        results_devel, results1l_devel = test(args, model1, device, devel_loader, model1l, devel_loader1l)
        results_test, results1l_test = test(args, model1, device, test_loader, model1l, test_loader1l)

        if float(results_devel['uar']) > float(best_results_devel['uar']):
            best_results_devel = results_devel
            best_restuls_test = results_test
            torch.save(model1.state_dict(), './modelLing.pt')

        if float(results1l_devel['uar']) > float(best_results1l_devel['uar']):
            best_results1l_devel = results1l_devel
            best_restuls1l_test = results1l_test
            torch.save(model1l.state_dict(), './modelSyll.pt')

    # print(model1l.state_dict())

    return best_results_devel, best_restuls_test, best_results1l_devel, best_restuls1l_test


if __name__ == '__main__':
    # main()
    best_results_devel, best_restuls_test, best_results1l_devel, best_restuls1l_test = main()

    print('LING: uar_devel: {}\t war_devel: {}\t uar_test: {}\t war_test: {}'.format(
        best_results_devel['uar'], best_results_devel['war'], best_restuls_test['uar'], best_restuls_test['war']))

    print('SYLL: uar_devel: {}\t war_devel: {}\t uar_test: {}\t war_test: {}'.format(
        best_results1l_devel['uar'], best_results1l_devel['war'], best_restuls1l_test['uar'], best_restuls1l_test['war']))