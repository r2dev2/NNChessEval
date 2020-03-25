import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from Common import fenToInputs, evalSimplify


class ChessDataset(Dataset):
    def __init__(self, fenfile, evalfile):
        with open(fenfile, 'r') as fin:
            self.x_data = torch.Tensor([fenToInputs(fen[:-1]) for fen in fin.readlines()])
        with open(evalfile, 'r') as fin:
            self.y_data = torch.LongTensor([evalSimplify(e[:-1]) for e in fin.readlines()])
            # self.y_data = Variable(torch.Tensor([int(evalSimplify(e[:-1])[2] == 1) for e in fin.readlines()]))
        print(len([i for i in self.y_data if float(i) == 2]))
        print(len(self.y_data))
        self.len = min(len(self.x_data), len(self.y_data))
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def main():
    print(fenToInputs("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"))
    d = ChessDataset("data/testData/fen.txt", "data/testData/eval.txt")
    with open("testloader.pickle", 'wb+') as fout:
        pickle.dump(d, fout)
    print(d[3])

if __name__ == "__main__":
    main()

