import numpy as np


def match_3_pos(c):
    if isinstance(c, np.ndarray):
        c = c.tolist()
    t = [1, 1, 1]
    return c == t


def match_3_neg(c):
    if isinstance(c, np.ndarray):
        c = c.tolist()
    t = [1, 1, 1]
    return c == t * (-1)


def match_3(c):
    return match_3_pos(c) or match_3_neg(c)


class Board:
    def __init__(self, board=None):
        if board is None:
            board = np.zeros([15, 15])
        self.board = board

    def is_game_end(self) -> bool:
        return self.is_pos_win() or self.is_neg_win()

    def is_pos_win(self):
        s = self.board
        for i in range(3):
            if match_3_pos(s[i]):
                return True
        for i in range(3):
            if match_3_pos(s[:, i].transpose([1, 0])):
                return True
        if match_3_pos((s[0, 0], s[1, 1], s[2, 2])):
            return True
        if match_3_pos((s[0, 2], s[1, 1], s[2, 0])):
            return True
        return False

    def is_neg_win(self):
        s = self.board
        for i in range(3):
            if match_3_neg(s[i]):
                return True
        for i in range(3):
            if match_3_neg(s[:, i].transpose([1, 0])):
                return True
        if match_3_neg((s[0, 0], s[1, 1], s[2, 2])):
            return True
        if match_3_neg((s[0, 2], s[1, 1], s[2, 0])):
            return True
        return False

    def reverse(self):
        result = Board()
        ...


def reward(s: Board) -> {-1, 1, 0}:
    """以黑色为视角，赢为1，输为-1"""
    if s.is_pos_win():
        return 1
    elif s.is_neg_win():
        return -1
    else:
        return 0


def get_valid_actions(s: Board) -> list:
    ...


def normalize_actions_probability(p, valid_actions) -> dict:
    """(if not already win)
    剔除不合理数据后对合理概率归一化"""
    ...


def next_state(s: Board, a) -> Board:
    """returns board after move 'a'"""
    ...


class Game:
    def __init__(self):
        """set up the board"""
        ...

    def move(self, a):
        """update the board after action 'a'"""
        ...
