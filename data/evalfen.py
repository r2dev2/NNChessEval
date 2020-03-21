import sys
from threading import Thread

import chess
import chess.engine

class threadWriter:
    def __init__(self, f):
        self.fout = open(f, 'a+')
        self.contents = ''
    def write(self, msg):
        self.contents += msg
    def close(self):
        self.fout.close()
        del self
    def flush(self):
        self.fout.write(self.contents)
        self.fout.flush()
    def read(self, no):
        pass
    def readlines(self):
        pass

def evalFENThread(output, i, fen, engine, d):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=d))
    output[i] = str(info["score"].white()) + '\n'

def main(filein, fileout, d, threads):
    engines = [chess.engine.SimpleEngine.popen_uci("stockfish") for i in range(threads)]

    with open(filein, 'r') as fin:
        contents = fin.readlines()

    counter = 0
    with open(fileout, 'a+') as fout:
        l = len(contents)
        while counter < l:
            ts = []
            threadcontents = [0 for i in range(threads)]
            for i in range(threads):
                try:
                    f = contents.pop(0)
                    fen = f[:-1]
                    t = Thread(target = evalFENThread, args = (threadcontents, i, fen, engines[i], d))
                    ts.append(t)
                except IndexError:
                    print("Almost done")
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            for c in threadcontents:
                try:
                    fout.write(c)
                    fout.flush()
                except:
                    if c != 0:
                        print("Add this:", c)
            counter += threads
            print(counter)
            
    print("Done")
    for engine in engines: engine.quit()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

