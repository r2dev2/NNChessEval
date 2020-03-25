from itertools import count
import pickle
import sys

from torch.autograd import Variable
import torch
import torch.nn.functional as F

from Loader import fenToInputs, evalSimplify, ChessDataset
from Model import Model
import argparse

OTHERS = {
    0: (1,2),
    1: (0,2),
    2: (0,1)
}

def test(model, fen, needstoparse = True):
    if needstoparse:
        listfen = fenToInputs(fen)
        v = torch.Tensor([listfen])
    else:
        v = fen
    with torch.no_grad():
        output = model(v)
    pred = output.data.max(1, keepdim=True)[-1]
    return pred
    # return F.softmax(model(Variable(torch.Tensor(listfen))))

def main(interactive = True):
    
    if args.model.split(".")[-1] == "pickle":
        with open(args.model, 'rb') as fin:
            model = pickle.load(fin)
        model.eval()
    else:
        model = Model()
        model.load_state_dict(torch.load(args.model))
        model.eval()
    
    if interactive:
        while True:
            print(test(model, input("fen? ")))
    else:
        # data/testData/fen.txt, eval.txt
        with open("data/testData/fen.txt", 'r') as fin:
            fens = [s[:-1] for s in fin.readlines()]
        with open("data/testData/eval.txt", 'r') as fin:
            ev = [s[:-1] for s in fin.readlines()]
        print(f"Evaluating on {len(fens)} samples")
        evals = (evalSimplify(e) for e in ev)
        correct = 0
        corrects = [0] * 3
        incorrects = [[], [], []] 
        total = 0
        totals = [0] * 3
        for f, e in zip(fens, evals):
            pred = test(model, f)
            with open("out.log", 'a+') as fout:
                print(pred, e, file = fout)
            if str(e) in str(pred):
                correct += 1
                corrects[e] += 1
            else:
                incorrects[e].append(str(pred)[9])
            # if str(e-1) in str(pred) or str(e+1) in str(pred):
            #     if "--lenient" in sys.argv:
            #         correct += 1
            total += 1
            totals[e] += 1
            if total % 1000 == 0:
                print(f"{correct} correct out of {total} samples, accuracy {correct/total}")
        print(f"num_incorrect: {len(incorrects)}")
        print(f"Finished: {correct} correct out {total} samples, accuracy {correct/total}")
        for i, c, ic, t in zip(count(), corrects, incorrects, totals):
            os = OTHERS[i]
            oc1 = ic.count(str(os[0]))
            oc2 = ic.count(str(os[1]))
            print(len(ic))
            print(i, c, t, c/t, oc1/len(ic), oc2/len(ic), sep = '\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", default="model.pt", help="path to pretrained model")
    parser.add_argument("-i", "--interactive", default=False, help="let user specify fens to eval")
    args = parser.parse_args()
    
    main(args.interactive)

