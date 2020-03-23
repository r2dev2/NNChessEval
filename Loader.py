import pickle

import torch
from torch.utils.data import Dataset, DataLoader

LETTER_TO_NUMBER = {
        'k' : -4,
        'q' : -9,
        'r' : -5,
        'b' : -3.25,
        'n' : -3,
        'p' : -1,
        'P' : 1,
        'N' : 3,
        'B' : 3.25,
        'R' : 5,
        'Q' : 9,
        'K' : 4
}

def castlingRights(fen):
    endpart = ''.join(fen.split(' ')[1:])
    rights = [
        'K' in endpart,
        'Q' in endpart,
        'k' in endpart,
        'q' in endpart
    ]
    return [int(b) for b in rights]

# rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
def fenToInputs(fen, normalize = True):
    output = []
    ranks = fen.split('/')
    last = ranks.pop().split(' ')
    ranks.append(last.pop(0))
    color = last.pop(0)
    for rank in ranks:
        for c in rank:
            if c in LETTER_TO_NUMBER:
                toadd = LETTER_TO_NUMBER[c]
                if normalize:
                    toadd /= 5
                output.append(toadd)
            else:
                for i in range(int(c)): output.append(0)
    output.append(int(color == 'w'))
    rights = castlingRights(fen)
    return output + rights

# 0 is black win, 2 is white win
def evalSimplify(ipt):
    try:
        e = eval(ipt)
    except:
        if ipt[1] == '-':
            return 0
        return 2
    if e <= -1.5 * 100:
        return 0
    elif -1.5 * 100 < e <= -.75 * 100:
        return 0
    elif -.75 * 100 < e < .75 * 100:
        return 1
    elif .75 * 100 <= e < 1.5 * 100:
        return 2
    else:
        return 2

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

