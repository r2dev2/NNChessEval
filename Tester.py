import pickle
import sys

from torch.autograd import Variable
import torch
import torch.nn.functional as F

from Loader import fenToInputs, evalSimplify, ChessDataset
from Model import Model

def test(model, fen, needstoparse = True):
    if needstoparse:
        listfen = [fenToInputs(fen)]
        v = torch.Tensor([fenToInputs(fen)])
    else:
        v = fen
    output = model(v)
    pred = output.data.max(1, keepdim=True)[-1]
    return pred
    # return F.softmax(model(Variable(torch.Tensor(listfen))))

def main(interactive = True):
    with open(sys.argv[1], 'rb') as fin:
        model = pickle.load(fin)
    if interactive:
        while True:
            print(test(model, input("fen? ")))
    else:
        with open("data/testData/fen.txt", 'r') as fin:
            fens = [s[:-1] for s in fin.readlines()]
        with open("data/testData/fen.txt", 'r') as fin:
            evals = [s[:-1] for s in fin.readlines()]
        evals = (evalSimplify(e).index(1) for e in evals)
        correct = 0
        total = 0
        for f, e in zip(fens, evals):
            pred = test(model, f)
            if str(e) in str(pred):
                correct += 1
            # if str(e-1) in str(pred) or str(e+1) in str(pred):
            #     correct += .5
            total += 1
            if total % 1000 == 0:
                print(correct, total, correct/total)
        print(correct, total, correct/total)

if __name__ == "__main__":
    main('-t' not in sys.argv)

