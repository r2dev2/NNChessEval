from Bitboard import Bitboard

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
    board = Bitboard(fen)
    toreturn = board.to_np()
    del board
    return toreturn
    
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

if __name__ == "__main__":
    print(fenToInputs("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
