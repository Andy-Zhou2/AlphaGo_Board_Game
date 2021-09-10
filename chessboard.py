from random import randint, choice
import numpy

class ChessBoard:
    def __init__(self):
        self.board = [[0]*15 for _ in range(15)]
        # board: 0 for nothing, 1 for black, -1 for white

    def print_board(self):
        horizontal_line_length = 61
        self.str_board = [['●' if self.board[i][j]==1 else '○' if self.board[i][j]==-1 else ' ' for j in range(15)] for i in range(15)]
        for i in range(15):
            print('-'*horizontal_line_length)
            print('| ', end='')
            print(*self.str_board[i], sep=' | ', end=' |\n')
        print('-'*horizontal_line_length)

    def move(self, x, y, player):
        '''
        :param x: x coord of move, from 0 to 14
        :param y: y coord of move, from 0 to 14
        :param player: 1 for black, -1 for white
        :return:
        '''
        assert player == 1 or player == -1
        assert 0 <= x <= 14 and 0 <= y <= 14
        self.board[x][y] = player

    def validate_board(self):
        '''
        validates the current status of the board
        does so by putting masks on double 3s and double 4s
        :return:
        '''

        mask_double_3 = [[0]*15 for _ in range(15)]
        mask_double_4 = [[0]*15 for _ in range(15)]

        def b(x,y):
            return self.board[x][y]

        # 活双三
        for row in range(15):
            for s in range(15-4):
                if b(row,s+1)==1 and b(row,s+2)==1 and b(row, s+3)==1 \
                        and b(row, s)==0 and b(row, s+4)==0:
                    mask_double_3[row][s+1] = mask_double_3[row][s+2] = mask_double_3[row][s+3] = 1
        for col in range(15):
            for s in range(15-4):
                if self.board[s+1][col]==1 and self.board[s+2][col]==1 and self.board[s+3][col]==1 \
                        and self.board[s][col]==0 and self.board[s+4][col]==0:

                    mask_double_3[s+1][col] = mask_double_3[s+2][col] = mask_double_3[s+3][col] = 1

        # 双四
        # 边上：要求至少一个是可以落子的
        for row in range(15):
            for s in range(15-3):
                # 先判断边角
                # 3种情况：2个边角和其它
                if s == 0:
                    if b(row, 4)==1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False
                elif s == 11:
                    if b(row, 10)==1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False
                else:
                    if b(row, s-1)==1 or b(row, s+4)==1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False

                if b(row, s)==1 and b(row, s+1)==1 and b(row, s+2)==1 and b(row, s+3)==1 and boundary_ok_flag:
                    mask_double_4[row][s] = mask_double_4[row][s+1] = mask_double_4[row][s+2] = mask_double_4[row][s+3] = 1
        for col in range(15):
            for s in range(15-3):
                if s == 0:
                    if b(4, col) == 1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False
                elif s == 11:
                    if b(10, col) == 1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False
                else:
                    if b(s - 1, col) == 1 or b(s + 4, col) == 1:
                        boundary_ok_flag = True
                    else:
                        boundary_ok_flag = False

                if b(s, col) == 1 and b(s + 1, col) == 1 and b(s + 2, col) == 1 and b(s + 3, col) == 1 and boundary_ok_flag:
                    mask_double_4[s][col] = mask_double_4[s+1][col] = mask_double_4[s+2][col] = mask_double_4[s+3][col] = 1

        # 长连
        for row in range(15):
            count = 0
            for col in range(15):
                if b(row, col) == 1:
                    count += 1
                    if count > 5:
                        return False


board = ChessBoard()
board.move(3,3,1)
board.move(randint(0,14),randint(0,14),choice([-1,1]))
board.move(randint(0,14),randint(0,14),choice([-1,1]))
board.print_board()