import pickle
import sys

#import adabound
import torch
#from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

from Loader import ChessDataset, fenToInputs

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l1 = torch.nn.Linear(69, 50)
        self.l2 = torch.nn.Linear(50, 30)
        self.l3 = torch.nn.Linear(30, 20)
        self.l4 = torch.nn.Linear(20, 10)
        self.l5 = torch.nn.Linear(10, 3)
        
        self.activation = torch.tanh
        # self.activation = torch.nn.LeakyReLU(.2, inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.activation(self.l1(x))
        out = self.activation(self.l2(out))
        out = self.activation(self.l3(out))
        out = self.activation(self.l4(out))
        out = self.l5(out)
        # gfn = out.grad_fn
        # new = []
        # for i, t in enumerate(out):
        #     p = float(t)
        #     if p <= -.8:
        #         y_pred = -2
        #     elif -.8 < p <= -.3:
        #         y_pred = -1
        #     elif -.3 < p < .3:
        #         y_pred = 0
        #     elif .3 <= p < .8:
        #         y_pred = 1
        #     else:
        #         y_pred = 2
        #     out[i] = torch.FloatTensor([y_pred])
        #     print(out)
        #     new.append(y_pred)
        # # return Variable(torch.Tensor([[y] for y in new]))
        # out.grad_fn = gfn
        # print(out)
        return F.softmax(out, dim = 1)

def main(train = True, lr=1e-3):
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
    
    modelpath = "model.pt"
    
    if train:
        LOSS = []
        model = Model()
        
        try:
            f = open(modelpath, 'rb')
            f.close()
            model.load_state_dict(torch.load(modelpath))
        except:
            pass

    
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=.9)
        # optimizer = adabound.AdaBound(model.parameters(), lr = .001, final_lr = .1)


        print("Starting to train, good luck")
        for epoch in range(100):
            model.train()
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(data)
                loss = criterion(y_pred, target)
                if i % 1000 == 0: print(epoch, i, loss.item())

                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                for fen in fens: 
                    output = model(torch.Tensor([fenToInputs(fen)]))
                    pred = output.data.max(1, keepdim=True)[1]
                    print(pred)
            print(epoch, i, loss.item())
            LOSS.append(str(loss.item()))
            # if any([f in str(loss) for f in ('0.0', '0.01', '0.02', '0.03', '0.04')]): 
            #     print("Maybe, this is the one")
            #     modelpath = "yeetmodel.pt"
            #     break
        for fen in fens: print(model(torch.Tensor([fenToInputs(fen)])), fen)
        print(LOSS)
        torch.save(model.state_dict(), modelpath)
    else:
        model = Model()
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        for fen in fens:
            output = model(torch.Tensor([fenToInputs(fen)]))
            pred = output.data.max(1, keepdim=True)[1]
            print(f"pred: {pred.item()} correct: {pred == torch.Tensor([[1]])} {fen}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="to train or not to train", default=True)
    parser.add_argument("-lr", help="learning rate for training", default = 1e-3)
    args = parser.parse_args()
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
    main(args.t, args.lr)

