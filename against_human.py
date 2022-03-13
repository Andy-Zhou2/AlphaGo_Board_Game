import os
from model import GoBangNet
from TreeSearch import TreeSearch
from gobang_board import GoBangBoard, Player, index_to_coord, coord_to_index
import numpy as np


net_path = './data/nets'
net_filename = f"gen_{13280}.net"

search_num = 2000

human_plays_black = 0

net = GoBangNet().cuda()
net.load_param(os.path.join(net_path, net_filename))


# 10 games with net1 as black


tree = TreeSearch(net)

game_now = GoBangBoard()
player_now = Player.BLACK

def get_xy_input():
    while True:
        try:
            x, y = input("Input x,y: ").split(',')
            x, y = int(x), int(y)
            if 0 <= x < 15 and 0 <= y < 15:
                if game_now.get_valid_actions()[coord_to_index([x, y])] == 1:
                    return x, y
        except:
            print("Invalid input")
        else:
            print("Invalid input")

game_now.print_board()
while not game_now.is_game_ended():
    if (player_now == Player.BLACK and human_plays_black) or (player_now == Player.WHITE and not human_plays_black):
        x, y = get_xy_input()

        move = coord_to_index([x, y])
        game_now = game_now.move(move)
        player_now = Player.WHITE if player_now == Player.BLACK else Player.BLACK
        game_now.print_board()
    else:
        print('computer thinking...')
        policy, value = net.predict(game_now)
        print('computer thinks that win rate for itself is:', value)
        print('top policies:')
        ind = np.argpartition(policy, -4)[-4:]
        for i in reversed(ind[np.argsort(policy[ind])]):
            print(f'{index_to_coord(i)}: {policy[i]}')
        tree.search(game_now)
        # tree.add_noise(game_now, 0.1)
        for search_count in range(search_num):
            tree.search(game_now)
        _, move = tree.get_pi_and_get_move(0, game_now)
        game_now = game_now.move(move)
        player_now = Player.WHITE if player_now == Player.BLACK else Player.BLACK
        game_now.print_board()
        print('move: ', index_to_coord(move))
        policy, value = net.predict(game_now)
        print('computer thinks that win rate for you is:', value)

if game_now.is_black_win():
    print("Black wins")
else:
    print("White wins")

