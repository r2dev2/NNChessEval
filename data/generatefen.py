import sys

import chess.pgn

def getLength(filename):
    with open(filename, 'r') as fin:
        contents = fin.readlines()
    return len(''.join([s for s in contents if '[' not in s]).split('\n\n'))

def main(filein, fileout):
    l = getLength(filein)
    pgn = open(filein, 'r')
    games = [chess.pgn.read_game(pgn) for i in range(l)]

    with open(fileout, 'a+') as fout:
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                print(repr(board)[7:-2], file = fout)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

