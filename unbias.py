from itertools import count
import sys

import Loader

# removes duplicates in 1st list
# returns l1 and l2 without the duplicate entries in l3
def removeDuplicates(l1, l2, l3):
    seen = set()
    fl1, fl2 = [], []
    for i, l, m, n in zip(count(), l1, l2, l3):
        if tuple(n) not in seen:
            fl1.append(l)
            fl2.append(m)
            seen.add(tuple(l))
    del seen
    return fl1, fl2

# returns the two lists if the 3rd is unbiased
def unbias(l1, l2, l3):
    target = min([l3.count(i) for i in range(3)])
    counters = [0] * 3
    fl1, fl2 = [], []
    for i, l, m, n in zip(count(), l1, l2, l3):
        if counters[n] <= target:
            fl1.append(l)
            fl2.append(m)
            counters[n] += 1
    return fl1, fl2

def main(fenin, evalin, fenout, evalout):
    with open(fenin, 'r') as fin:
        fen = [s[:-1] for s in fin.readlines()]
    with open(evalin, 'r') as fin:
        streval = [s[:-1] for s in fin.readlines()]
    lfens = [Loader.fenToInputs(f) for f in fen]
    fen, streval = removeDuplicates(fen, streval, lfens)
    evals = [Loader.evalSimplify(e) for e in streval]
    fen, streval = unbias(fen, streval, evals)
    fout = open(fenout, 'a+')
    eout = open(evalout, 'a+')
    for f, e in zip(fen, streval):
        print(f, file = fout)
        print(e, file = eout)
    fout.close()
    eout.close()

if __name__ == "__main__":
    if len(sys.argv) == 1 or '-h' in sys.argv:
        print("run with options:")
        print("fenin evalin fenout evalout")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

