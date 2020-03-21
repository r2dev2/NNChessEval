import pickle
import sys

from torch.autograd import Variable
import torch
import torch.nn.functional as F

from Loader import fenToInputs
from Model import Model

def test(model, fen):
    listfen = [fenToInputs(fen)]
    print(listfen)
    output = model(Variable(torch.Tensor([fenToInputs(fen)])))
    pred = output.data.max(1, keepdim=True)[1]
    return pred
    # return F.softmax(model(Variable(torch.Tensor(listfen))))

def main():
    with open(sys.argv[1], 'rb') as fin:
        model = pickle.load(fin)
    while True:
        print(test(model, input("fen? ")))

if __name__ == "__main__":
    main()

