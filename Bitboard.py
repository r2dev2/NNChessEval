from copy import deepcopy

EMPTY_BOARD = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# r1bqk2r/pp2ppbp/2np1np1/8/4P3/2N3P1/PPP1NPBP/R1BQK2R b KQkq - 4 8
class Bitboard:
    def __init__(self, fen):
        self.fen = fen
        self.wrook = deepcopy(EMPTY_BOARD)
        self.wnight = deepcopy(EMPTY_BOARD)
        self.wbish = deepcopy(EMPTY_BOARD)
        self.wqueen = deepcopy(EMPTY_BOARD)
        self.wking = deepcopy(EMPTY_BOARD)
        self.wpawn = deepcopy(EMPTY_BOARD)
        self.brook = deepcopy(EMPTY_BOARD)
        self.bnight = deepcopy(EMPTY_BOARD)
        self.bbish = deepcopy(EMPTY_BOARD)
        self.bqueen = deepcopy(EMPTY_BOARD)
        self.bking = deepcopy(EMPTY_BOARD)
        self.bpawn = deepcopy(EMPTY_BOARD)

        piece_to_layer = {
            'R': self.wrook,
            'N': self.wnight,
            'B': self.wbish,
            'Q': self.wqueen,
            'K': self.wking,
            'P': self.wpawn,
            'p': self.bpawn,
            'k': self.bking,
            'q': self.bqueen,
            'b': self.bbish,
            'n': self.bnight,
            'r': self.brook
        }


        parts = self.fen.split(' ')
        ranks = parts[0].split('/')
        otherinfo = ''.join(parts[1:])
        self.infolayer = deepcopy(EMPTY_BOARD)

        self.color = int('w' in otherinfo)
        self.wkcastle = int('K' in otherinfo)
        self.wqcastle = int('Q' in otherinfo)
        self.bkcastle = int('k' in otherinfo)
        self.bqcastle = int('q' in otherinfo)

        self.infolayer[4] = 8 * [self.color]
        self.infolayer[0] = 8 * [self.wkcastle]
        self.infolayer[1] = 8 * [self.wqcastle]
        self.infolayer[6] = 8 * [self.bkcastle]
        self.infolayer[7] = 8 * [self.bqcastle]

        for i, rank in zip(range(7, -1, -1), ranks):
            fil = 0
            for j, s in enumerate(rank):
                if s in '12345678':
                    fil  += int(s)
                else:
                    piece_to_layer[s][i][fil] = 1
                    fil += 1

        
    def to_list(self):
        return [self.wrook, self.wnight, self.wbish, self.wqueen, self.wking, self.wpawn, 
                self.brook, self.bnight, self.bbish, self.bqueen, self.bking, self.bpawn,
                self.infolayer]

    def __str__(self):
        return self.fen


