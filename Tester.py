import pickle
import sys

from torch.autograd import Variable
import torch
import torch.nn.functional as F

from Loader import fenToInputs, evalSimplify, ChessDataset
from Model import Model

def test(model, fen, needstoparse = True):
    if needstoparse:
        listfen = fenToInputs(fen)
        v = torch.Tensor([listfen])
    else:
        v = fen
    output = model(v)
    pred = output.data.max(1, keepdim=True)[-1]
    return pred
    # return F.softmax(model(Variable(torch.Tensor(listfen))))

def main(interactive = True):
    if '--legacy' not in sys.argv:
        model = Model()
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
    else:
        with open(sys.argv[1], 'rb') as fin:
            model = pickle.load(fin)
        model.eval()
    if interactive:
        while True:
            print(test(model, input("fen? ")))
    else:
        with open("data/testData/fen.txt", 'r') as fin:
            fens = [s[:-1] for s in fin.readlines()]
        with open("data/testData/eval.txt", 'r') as fin:
            ev = [s[:-1] for s in fin.readlines()]
        evals = (evalSimplify(e) for e in ev)
        correct = 0
        total = 0
        for f, e in zip(fens, evals):
            pred = test(model, f)
            with open("out.log", 'a+') as fout:
                print(pred, e, file = fout)
            if str(e) in str(pred):
                correct += 1
            # if str(e-1) in str(pred) or str(e+1) in str(pred):
            #     if "--lenient" in sys.argv:
            #         correct += 1
            total += 1
            if total % 1000 == 0:
                print(correct, total, correct/total)
        print(correct, total, correct/total)

if __name__ == "__main__":
    main('-t' not in sys.argv)

