import numpy as np
from enum import Enum
import torch


# BLACK_TO_MOVE = np.ones([15, 15])  # represents that black should move now
# WHITE_TO_MOVE = np.zeros([15, 15])  # represents that white should move now


class Player(Enum):
    BLACK = 1
    WHITE = 0


def another_player(player):
    return Player.BLACK if player == Player.WHITE else Player.WHITE


def check_win_single_board(board):
    """
    :return: True if there is a win on the board
    """
    n = 5
    for w in range(15):
        for h in range(15):
            if (w in range(15 - n + 1) and board[w][h] != 0 and
                    len(set(board[i][h] for i in range(w, w + n))) == 1):
                return True
            if (h in range(15 - n + 1) and board[w][h] != 0 and
                    len(set(board[w][j] for j in range(h, h + n))) == 1):
                return True
            if (w in range(15 - n + 1) and h in range(15 - n + 1) and board[w][h] != 0 and
                    len(set(board[w + k][h + k] for k in range(n))) == 1):
                return True
            if (w in range(15 - n + 1) and h in range(n - 1, 15) and board[w][h] != 0 and
                    len(set(board[w + l][h - l] for l in range(n))) == 1):
                return True
    return False


def index_to_coord(index):
    """
    used when converting index of p from the network to (row, col) pair
    :param index:
    :return:
    """
    return index // 15, index % 15


def coord_to_index(coord):
    row, col = coord
    return row * 15 + col


def normalize_actions_probability(p, valid_actions_mask) -> np.array:
    """
    (assume not already win)
    剔除不合理数据后对合理概率归一化
    note: p是原始概率
    valid actions: a mask of valid actions
    """
    p = p * valid_actions_mask
    p = p / p.sum()
    return p


def get_symmetries(board, pi):
    """
    adapted from https://github.com/suragnair/alpha-zero-general
    given board and pi distribution, generate symmetries
    """
    pi_board = np.reshape(pi, (15, 15))
    result = []
    black_board, white_board, turn_board = board

    for i in range(1, 5):
        for j in [True, False]:
            newB = np.rot90(black_board, i)
            newW = np.rot90(white_board, i)
            newPi = np.rot90(pi_board, i)
            if j:
                newB = np.fliplr(newB)
                newW = np.fliplr(newW)
                newPi = np.fliplr(newPi)
            result.append([(newB, newW, turn_board), newPi.ravel()])
    return result


class GoBangBoard:
    def __init__(self, information=None):
        if information is None:
            self.next = np.zeros([15, 15])
            self.prev = np.zeros([15, 15])
            self.next_player = Player.BLACK
            self.last_move = None  # stores last move in index form
        else:
            self.next, self.prev, self.next_player, self.last_move = information

    def get_str_representation(self):
        """
        :return: string representation of the board
        """
        return np.concatenate([self.next, self.prev]).tostring()

    def print_board(self):
        horizontal_line_length = 61
        black = self.next if self.next_player == Player.BLACK else self.prev
        white = self.prev if self.next_player == Player.BLACK else self.next
        str_board = [['●' if black[i][j] else '○' if white[i][j] else ' ' for j in range(15)]
                     for i in range(15)]
        if self.last_move is not None:
            x, y = index_to_coord(self.last_move)
            str_board[x][y] = '△' if self.next_player == Player.BLACK else '▲' \
                # if next player is black then last move is white
        for i in range(15):
            print('-' * horizontal_line_length)
            print('| ', end='')
            print(*str_board[i], sep=' | ', end=' |\n')
        print('-' * horizontal_line_length)

    def move(self, a):
        """
        places a stone on the board
        does not check for wins
        a: the move, 0 <= a < 225
        :return: a new board object, turn reversed
        """
        assert 0 <= a < 225
        x, y = index_to_coord(a)

        if self.next[x][y] == 1 or self.prev[x][y] == 1:
            print('error - trying to place a stone on an occupied position!')
            print(x, y)
            self.print_board()
            raise ValueError("This position is already occupied")

        new_board = GoBangBoard((self.prev.copy(), self.next.copy(), another_player(self.next_player), a))
        new_board.prev[x][y] = 1
        return new_board

    def is_game_ended(self):
        """
        checks if the game ended: any win, or cannot place a stone
        :return: True if game ended
        """
        if check_win_single_board(self.prev) or check_win_single_board(self.next):
            return True
        if np.sum(self.prev + self.next) == 225:
            return True
        return False

    def is_black_win(self):
        """
        checks if black won
        :return: True if black won
        """
        return check_win_single_board(self.next if self.next_player == Player.BLACK else self.prev)

    def is_white_win(self):
        """
        checks if white won
        :return: True if white won
        """
        return check_win_single_board(self.next if self.next_player == Player.WHITE else self.prev)

    def get_valid_actions(self):
        """
        returns a list of valid actions
        :return: a list of valid actions
        """
        valid_actions = np.ones([15, 15])

        valid_actions[self.next == 1] = 0
        valid_actions[self.prev == 1] = 0

        # print(valid_actions.tolist())

        return valid_actions.reshape([225])

    def get_reward(self):
        """
        returns reward for the current board, black wins then 1, white then -1, otherwise 0
        :return: reward
        """
        if self.is_black_win():
            return 1 if self.next_player == Player.BLACK else -1
        elif self.is_white_win():
            return 1 if self.next_player == Player.WHITE else -1
        else:
            return 0

    def get_network_input(self):
        """
        :return: array of [next, prev, last move]
        """
        last_move_plane = np.zeros([15, 15])
        if self.last_move is not None:
            x, y = index_to_coord(self.last_move)
            last_move_plane[x][y] = 1
        return torch.Tensor(np.array([self.next, self.prev, last_move_plane])).unsqueeze(0)
