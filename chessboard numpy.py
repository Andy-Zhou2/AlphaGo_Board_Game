from random import randint, choice
import numpy as np


class ChessBoard:
    def __init__(self):
        self.board = np.zeros([15, 15])
        # board: 0 for nothing, 1 for black, -1 for white

    def print_board(self):
        horizontal_line_length = 61
        str_board = [['●' if self.board[i][j] == 1 else '○' if self.board[i][j] == -1 else ' ' for j in range(15)]
                          for i in range(15)]
        for i in range(15):
            print('-' * horizontal_line_length)
            print('| ', end='')
            print(*str_board[i], sep=' | ', end=' |\n')
        print('-' * horizontal_line_length)

    def move(self, x, y, player):
        '''
        places a stone on the board
        does not check for wins
        :param x: x coord of move, from 0 to 14
        :param y: y coord of move, from 0 to 14
        :param player: 1 for black, -1 for white
        :return:
        '''
        assert player == 1 or player == -1
        assert 0 <= x <= 14 and 0 <= y <= 14
        if self.board[x][y] != 0:
            return False

        self.board[x][y] = player
        if not self.validate_board():
            self.board[x][y] = 0  # reverse change
            return False
        return True

    def check_long_connection_horizontal(self, board, length, target):
        '''
        checks horizontally if there is any connection of length "length"
        returns True if such connection found
        :param board: board to be checked
        :param length: check length
        :param target: connection type (e.g. 1 or -1)
        :return:
        '''
        for row in range(15):
            for s in range(15 - length + 1):  # max: 9 10 11 12 13 14 checked
                if board[row][s:s + length].tolist() == [target] * length:
                    return True
        return False

    def validate_board(self):
        '''
        validates the current status of the board
        does so by putting masks on double 3s and double 4s
        :return:
        '''

        # 活双三
        def get_mask_double_3_horizontal(board):
            mask = np.zeros([15, 15])
            for row in range(15):
                for s in range(11):  # max: 10 11 12 13 14 checked
                    if board[row][s:s + 5].tolist() == [0, 1, 1, 1, 0]:
                        mask[row][s + 1:s + 4] = 1
            return mask

        mask_double_3_horizontal = get_mask_double_3_horizontal(self.board)
        mask_double_3_vertical = get_mask_double_3_horizontal(self.board.transpose(1, 0))

        if np.sum(mask_double_3_horizontal * mask_double_3_vertical):
            return False

        # 双四
        # 边上：要求至少一个是可以落子的
        def get_mask_double_4_horizontal(board):
            mask = np.zeros([15, 15])
            for row in range(15):
                for s in range(12):  # max: 11 12 13 14 checked
                    # 先判断边角
                    # 3种情况：2个边角和其它
                    if s == 0:
                        if board[row][4] == 1:
                            boundary_ok_flag = True
                        else:
                            boundary_ok_flag = False
                    elif s == 11:
                        if board[row][10] == 1:
                            boundary_ok_flag = True
                        else:
                            boundary_ok_flag = False
                    else:
                        if board[row][s - 1] == 1 or board[row][s + 4] == 1:
                            boundary_ok_flag = True
                        else:
                            boundary_ok_flag = False

                    if board[row][s:s + 4].tolist() == [1] * 4 and boundary_ok_flag:
                        mask[row][s:s + 4] = 1
            return mask

        mask_double_4_horizontal = get_mask_double_4_horizontal(self.board)
        mask_double_4_vertical = get_mask_double_4_horizontal(self.board.transpose(1, 0))

        if np.sum(mask_double_4_horizontal * mask_double_4_vertical):
            return False

        # 长连

        if self.check_long_connection_horizontal(self.board, 6, 1):
            return False
        if self.check_long_connection_horizontal(self.board.transpose(1, 0), 6, 1):
            return False

        return True

    def check_win(self):
        """
        returns 1 if black wins, -1 if white wins, 0 if neither wins
        :return:
        """
        if self.check_long_connection_horizontal(self.board, 5, 1) or \
                self.check_long_connection_horizontal(self.board.transpose(1, 0), 5, 1):
            return 1
        if self.check_long_connection_horizontal(self.board, 5, -1) or \
                self.check_long_connection_horizontal(self.board.transpose(1, 0), 5, -1):
            return -1
        return 0


a = np.array([1, 1, 1, 3, 4])

board = ChessBoard()
print(board.move(3, 3, -1))
print(board.move(3, 4, -1))
print(board.move(3, 5, -1))
print(board.move(3, 6, -1))
print(board.move(3, 7, -1))
board.move(randint(0, 14), randint(0, 14), choice([-1, 1]))
board.move(randint(0, 14), randint(0, 14), choice([-1, 1]))
print(board.check_win())
board.print_board()
