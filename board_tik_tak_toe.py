import numpy as np


def match_3(c):
    if isinstance(c, np.ndarray):
        c = c.tolist()
    t = [1, 1, 1]
    return c == t


def is_win(p):
    for i in range(3):
        if match_3(p[i]):
            return True
    for i in range(3):
        if match_3(p[:, i]):
            return True
    if match_3((p[0, 0], p[1, 1], p[2, 2])):
        return True
    if match_3((p[0, 2], p[1, 1], p[2, 0])):
        return True
    return False


class Board:
    def __init__(self, boards=None):
        if boards is None:
            self.black = np.zeros([3, 3])
            self.white = np.zeros([3, 3])
        else:
            self.black, self.white = boards

    def is_game_end(self) -> bool:
        return self.is_black_win() or self.is_white_win()

    def is_black_win(self):
        return is_win(self.black)

    def is_white_win(self):
        return is_win(self.white)

    def reverse(self):
        """returns a new board"""
        return reverse(self)

    def print_self(self):
        horizontal_line_length = 13
        str_board = [
            ['●' if self.black[i][j] else '○' if self.white[i][j] else ' ' for j in range(3)] for i in
            range(3)]
        for i in range(3):
            print('-' * horizontal_line_length)
            print('| ', end='')
            print(*str_board[i], sep=' | ', end=' |\n')
        print('-' * horizontal_line_length)


def reverse(s: Board):
    return Board([s.white, s.black])


def reward(s: Board) -> {-1, 1, 0}:
    """以黑色为视角，赢为1，输为-1"""
    if s.is_black_win():
        return 1
    elif s.is_white_win():
        return -1
    else:
        return 0


def get_all_actions() -> list:
    result = []
    for row in range(3):
        for col in range(3):
            result.append((row, col))
    return result


def get_valid_actions(s: Board) -> list:
    """
    唯一规则：已经有的地方不能再落子
    """
    result = []
    for a in get_all_actions():
        row, col = a
        if not (s.black[row][col] or s.white[row][col]):
            result.append(a)
    return result


def index_to_coord(index):
    """
    used when converting index of p from the network to (row, col) pair
    :param index:
    :return:
    """
    return index // 3, index % 3


def coord_to_index(coord):
    row, col = coord
    return row * 3 + col


def normalize_actions_probability(p, valid_actions) -> dict:
    """
    (if not already win)
    剔除不合理数据后对合理概率归一化
    note: p是原始概率
    """
    result = dict()
    sum_valid_pr = 0

    for a in valid_actions:
        sum_valid_pr += p[coord_to_index(a)]

    for a in valid_actions:
        result[a] = p[coord_to_index(a)] / sum_valid_pr

    return result


def next_state(s: Board, a) -> Board:
    """returns board after move 'a' (always assume it's black's turn)"""
    result = Board([s.black, s.white])
    row, col = a
    result.black[row][col] = 1
    return result
