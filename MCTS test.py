from MCTS import *

black_board = np.zeros((15, 15))
black_board[0, 0:4] = 1
white_board = np.zeros((15, 15))
turn = np.ones((15, 15))

from model import GoBangNet
net = GoBangNet().cuda()

root = Node(GoBangBoard((black_board, white_board, turn)), net)
print(root.s.print_board())
# print(np.all(root.s.black == BLACK_TO_MOVE))
tree = Tree(net)
tree.root = root
# root.edges[(0, 4)].expand(root.s.move(0, 4), net)
# search(net, root.edges[(0, 4)].son)
tree.search_from_root(40)
print(tree.root.edges[(0, 4)].expanded)
print('end')