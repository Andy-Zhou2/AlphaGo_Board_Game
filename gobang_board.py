from random import randint, choice
import numpy as np
from enum import Enum

BLACK_TO_MOVE = np.ones([15, 15])  # represents that black should move now
WHITE_TO_MOVE = np.zeros([15, 15])  # represents that white should move now


class Player(Enum):
    BLACK = 1
    WHITE = 0


def check_long_connection_horizontal(board, length):
    """
    checks horizontally if there is any connection of length "length"
    returns True if such connection found
    :param board: the board to check
    :param length: the length of the connection
    :return:
    """
    for row in range(15):
        for s in range(15 - length + 1):  # max: 9 10 11 12 13 14 checked
            if np.all(board[row, s:s + length] == np.ones(length)):
                return True
    return False


def check_diagonal_connection_up_left_to_down_right(board, length):
    """
    checks if there is any connection of length "length" in the upper left diagonal
    returns True if such connection found
    :param board: the board to check
    :param length: the length of the connection
    :return:
    """
    for s in range(15 - length + 1):
        for t in range(15 - length + 1):
            for i in range(length):
                if board[s + i, t + i] != 1:
                    break
            else:
                return True
    return False


def check_diagonal_connection_down_left_to_up_right(board, length):
    """
    checks if there is any connection of length "length" in the lower left diagonal
    returns True if such connection found
    :param board: the board to check
    :param length: the length of the connection
    :return:
    """
    for s in range(15 - length + 1):  # row
        for t in range(length - 1, 15):  # col
            for i in range(length):
                if board[s + i, t - i] != 1:
                    break
            else:
                return True
    return False


def check_win_single_board(board):
    """
    :return: True if there is a win on the board
    """
    if check_long_connection_horizontal(board, 5):
        return True
    if check_long_connection_horizontal(board.T, 5):
        return True
    if check_diagonal_connection_up_left_to_down_right(board, 5):
        return True
    if check_diagonal_connection_down_left_to_up_right(board, 5):
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


def normalize_actions_probability(p, valid_actions) -> dict:
    """
    (assume not already win)
    剔除不合理数据后对合理概率归一化
    note: p是原始概率
    @ param valid_actions: coordinates, not indices
    """
    result = dict()
    sum_valid_pr = 0

    for a in valid_actions:
        sum_valid_pr += p[coord_to_index(a)]

    for a in valid_actions:
        result[a] = p[coord_to_index(a)] / sum_valid_pr

    return result


class GoBangBoard:
    def __init__(self, boards=None):
        if boards is None:
            self.black = np.zeros([15, 15])
            self.white = np.zeros([15, 15])
            self.turn = BLACK_TO_MOVE
            self.current_player = Player.BLACK
        else:
            self.black, self.white, self.turn = boards
            if np.all(self.turn == BLACK_TO_MOVE):
                self.current_player = Player.BLACK
            else:
                self.current_player = Player.WHITE

    def get_str_representation(self):
        """
        :return: string representation of the board
        """
        return np.concatenate([self.black, self.white, self.turn]).tostring()

    def print_board(self):
        horizontal_line_length = 61
        str_board = [['●' if self.black[i][j] else '○' if self.white[i][j] else ' ' for j in range(15)]
                     for i in range(15)]
        for i in range(15):
            print('-' * horizontal_line_length)
            print('| ', end='')
            print(*str_board[i], sep=' | ', end=' |\n')
        print('-' * horizontal_line_length)

    def move(self, x, y):
        """
        places a stone on the board
        does not check for wins
        :param x: x coord of move, from 0 to 14
        :param y: y coord of move, from 0 to 14
        :return: a new board object, turn reversed
        """
        assert 0 <= x <= 14 and 0 <= y <= 14

        if self.black[x][y] == 1 or self.white[x][y] == 1:
            raise ValueError("This position is already occupied")

        new_board = GoBangBoard(boards=(self.black.copy(), self.white.copy(), self.turn.copy()))
        if self.current_player == Player.BLACK:
            new_board.black[x][y] = 1
            new_board.turn = WHITE_TO_MOVE
            new_board.current_player = Player.WHITE
        else:
            new_board.white[x][y] = 1
            new_board.turn = BLACK_TO_MOVE
            new_board.current_player = Player.BLACK
        return new_board

    def is_game_ended(self):
        """
        checks if the game ended: any win, or cannot place a stone
        :return: True if game ended
        """
        if check_win_single_board(self.black) or check_win_single_board(self.white):
            return True
        if np.sum(self.black + self.white) == 225:
            return True
        return False

    def is_black_win(self):
        """
        checks if black won
        :return: True if black won
        """
        return check_win_single_board(self.black)

    def is_white_win(self):
        """
        checks if white won
        :return: True if white won
        """
        return check_win_single_board(self.white)

    def get_valid_actions(self):
        """
        returns a list of valid actions
        :return: a list of valid actions
        """
        valid_actions = []
        for i in range(15):
            for j in range(15):
                if self.black[i][j] == 0 and self.white[i][j] == 0:
                    valid_actions.append((i, j))
        return valid_actions

    def reward(self):
        """
        returns reward for the current board, black wins then 1, white then -1, otherwise 0
        :return: reward
        """
        if self.is_black_win():
            return 1 if self.current_player == Player.BLACK else -1
        elif self.is_white_win():
            return -1 if self.current_player == Player.WHITE else 1
        else:
            return 0


if __name__ == '__main__':
    board = GoBangBoard()
    board = board.move(3, 3)
    board = board.move(4, 3)
    board = board.move_count(3, 4)
    board = board.move_count(4, 4)
    board = board.move_count(3, 5)
    board = board.move_count(4, 5)
    board = board.move_count(3, 6)
    board = board.move_count(4, 6)
    board = board.move_count(3, 7)

    board.print_board()

    print(board.is_game_ended())

    for i in range(100):
        try:
            board = board.move_count(randint(0, 14), randint(0, 14))
        except ValueError:
            pass
    print(board.is_game_ended())
    board.print_board()
