from Bitboard import Bitboard

l = Bitboard("r1bqk2r/pp2ppbp/2np1np1/8/4P3/2N3P1/PPP1NPBP/R1BQK2R b KQkq - 4 8").to_list()

for board in l:
    for row in board:
        print(row)
    print('\n')
