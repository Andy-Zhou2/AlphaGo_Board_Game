import os
from model import GoBangNet
from TreeSearch import TreeSearch
from gobang_board import GoBangBoard, Player

net_path = './data/nets'
net1_filename = "baseline.net"
net2_filename = "test_2_model_DNN.net"

net1 = GoBangNet().cuda()
net1.load_param(os.path.join(net_path, net1_filename))

net2 = GoBangNet().cuda()
net2.load_param(os.path.join(net_path, net2_filename))

net1_win = 0
net2_win = 0

# 10 games with net1 as black
# for i in range(10):
#     print("Game %d" % (i+1))
#
#     tree1 = TreeSearch(net1)
#     tree2 = TreeSearch(net2)
#
#     game_now = GoBangBoard()
#     player_now = Player.BLACK
#
#     while not game_now.is_game_ended():
#         if player_now == Player.BLACK:
#             for search_count in range(200):
#                 tree1.search(game_now)
#             _, move = tree1.get_pi_and_get_move(0, game_now)
#             game_now = game_now.move(move)
#             player_now = Player.WHITE
#         else:
#             for search_count in range(200):
#                 tree2.search(game_now)
#             _, move = tree2.get_pi_and_get_move(0, game_now)
#             game_now = game_now.move(move)
#             player_now = Player.BLACK
#         game_now.print_board()
#
#     if game_now.is_black_win():
#         print("Black wins")
#         net1_win += 1
#     else:
#         print("White wins")
#         net2_win += 1
#
#     game_now.print_board()

# 10 games with net1 as white
for i in range(10):
    print("Game %d" % (i+1))

    tree1 = TreeSearch(net2)
    tree2 = TreeSearch(net1)

    game_now = GoBangBoard()
    player_now = Player.BLACK

    while not game_now.is_game_ended():
        if player_now == Player.BLACK:
            for search_count in range(200):
                tree1.search(game_now)
            _, move = tree1.get_pi_and_get_move(0, game_now)
            game_now = game_now.move(move)
            player_now = Player.WHITE
        else:
            for search_count in range(200):
                tree2.search(game_now)
            _, move = tree2.get_pi_and_get_move(0, game_now)
            game_now = game_now.move(move)
            player_now = Player.BLACK
        game_now.print_board()

    if game_now.is_black_win():
        print("Black wins")
        net2_win += 1
    else:
        print("White wins")
        net1_win += 1

    game_now.print_board()


