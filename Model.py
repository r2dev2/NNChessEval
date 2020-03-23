import pickle
import sys

import adabound
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from Loader import ChessDataset, fenToInputs


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(13, 800, kernel_size = 4)
        self.conv2 = nn.Conv2d(800, 400, kernel_size = 1)
        self.mp = nn.MaxPool2d(2, stride = (2, 2))

        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        in_size = x.size(0)
        out = F.relu(self.mp(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = out.view(in_size, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return F.log_softmax(out, dim = 1)

def main(train = True):
    try:
        with open("dataloader.pickle", 'rb') as fin:
            train_loader = pickle.load(fin)
    except:
        chessdata = ChessDataset("data/unbiasfen.txt", "data/unbiaseval.txt")
        train_loader = DataLoader(
                dataset = chessdata,
                batch_size = 64,
                shuffle = True,
                num_workers = 4
        )
        with open("dataloader.pickle", 'wb+') as fout:
            pickle.dump(train_loader, fout)

    
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = .0002)
    
    if train:
        LOSS = []
        model = Model()
        modelpath = "model.pt"
        optimpath = "ada.pt"
        try:
            f = open(modelpath, 'rb')
            f.close()
            model.load_state_dict(torch.load(modelpath))
        except:
            pass


    
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = .001, momentum=.5)
        # optimizer = adabound.AdaBound(model.parameters(), lr = 1e-3, final_lr = .1)
        try:
            f = open(optimpath, 'rb')
            f.close()
            optimizer.load_state_dict(torch.load(optimpath))
        except:
            pass


        print("Starting to train, good luck")
        try:
            for epoch in range(50):
                    model.train()
                    for i, (data, target) in enumerate(train_loader):
                        data, target = Variable(data), Variable(target)
                        optimizer.zero_grad()
                        y_pred = model(data)
                        loss = criterion(y_pred, target)
                        if i % 1000 == 0: print(epoch, i, loss)

                        loss.backward()
                        optimizer.step()
                    if epoch % 10 == 0:
                        for fen in fens: 
                            output = model(torch.Tensor([fenToInputs(fen)]))
                            pred = output.data.max(1, keepdim=True)[1]
                            print(pred)
                    print(epoch, i, loss)
                    LOSS.append(str(loss))
                    # if any([f in str(loss) for f in ('0.0', '0.01', '0.02', '0.03', '0.04')]): 
                    #     print("Maybe, this is the one")
                    #     modelpath = "yeetmodel.pt"
                    #     break
        except KeyboardInterrupt:
            print("Saving Model")
        for fen in fens: print(model(Variable(torch.Tensor([fenToInputs(fen)]))), fen)
        print(LOSS)
        torch.save(model.state_dict(), modelpath)
        torch.save(optimizer.state_dict(), optimpath)
    else:
        model = Model()
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        for fen in fens:
            output = model(Variable(torch.Tensor([fenToInputs(fen)])))
            pred = output.data.max(1, keepdim=True)[1]
            print(pred, pred == torch.Tensor([[1]]), fen)


if __name__ == "__main__":
    if '-h' in sys.argv:
        print("use -t {learning rate} to train")
        print("starts training off model.pt unless it is removed")
        print("data is from dataloader.pickle unless it is removed")
        print("go into code to change filepaths of data")
        exit()
    fens = [
            "1rbq1rk1/3npp1p/2np2p1/3N4/1p1BP3/6PP/1PP2PB1/R2Q1RK1 b - - 0 16",
            "1rbq1rk1/3npp1p/2np2p1/3N4/1p1bP3/4B1PP/1PP2PB1/R2Q1RK1 w - - 0 16",
            "2bq1rk1/nr1n1p1p/3Q2p1/4p3/1p2P3/4N1PP/1PP2PB1/R4RK1 w - - 0 20",
            "rn1qkb1r/pbpp1ppp/1p2pn2/8/2PP4/P4N2/1P2PPPP/RNBQKB1R w KQkq - 1 5", # should be diff
            "rn1qk3/pbpp1ppp/1p2pn2/8/2PP4/P4N2/1P2PPPP/RNBQKB1R w KQq - 1 5",
            "r1bqk2r/pp2ppbp/2np1np1/8/4P3/2N3P1/PPP1NP1P/R1BQKB1R w KQkq - 3 8",
            "2bq2k1/nr1n1p1p/4r1p1/4p3/1N2P3/6PP/1PPQ1PB1/R4RK1 b - - 2 22",
            "2bq2k1/nr1n1p1p/4r1p1/4p3/1N2P3/6PP/1PPQ1PB1/R4RK1 w - - 2 22" # should be diff than prev
            ]
    main('-t' in sys.argv)

