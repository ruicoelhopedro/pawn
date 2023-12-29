import torch
import chess


def file(s):
    return s % 8


def rank(s):
    return s // 8


def make_square(r, f):
    return 8 * r + f


def horizontal_mirror(s):
    return make_square(rank(s), 7 - file(s))


def vm(s):
    return make_square(7 - rank(s), file(s))


def map_features(p, s, ks, t):
    if file(ks) >= 4:
        s = horizontal_mirror(s)
        ks = horizontal_mirror(ks)
    king_index = 4 * rank(ks) + file(ks)
    t = 0 if t else 1
    return s + 64 * ((p-1) + 5 * (king_index + 32 * t))


def build_features(fen: str):
    board = chess.Board(fen)
    kings = [board.pieces(chess.KING, a).pop() for a in (chess.WHITE, chess.BLACK)]
    features = [[], []]
    pieces = 0
    for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        pieces += len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK))
        features[0] += [map_features(p,    s ,    kings[0] , chess.WHITE) for s in board.pieces(p, chess.WHITE)]
        features[0] += [map_features(p,    s ,    kings[0] , chess.BLACK) for s in board.pieces(p, chess.BLACK)]
        features[1] += [map_features(p, vm(s), vm(kings[1]), chess.WHITE) for s in board.pieces(p, chess.BLACK)]
        features[1] += [map_features(p, vm(s), vm(kings[1]), chess.BLACK) for s in board.pieces(p, chess.WHITE)]
    return (
        torch.LongTensor([0]),
        torch.LongTensor(features[0]),
        torch.LongTensor([0]),
        torch.LongTensor(features[1]),
        torch.LongTensor([(pieces - 1) // 8]),
        torch.LongTensor([0 if board.turn == chess.WHITE else 1]),
    )
