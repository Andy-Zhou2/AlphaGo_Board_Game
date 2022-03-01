import os
from model import GoBangNet
from TreeSearch import TreeSearch
from gobang_board import GoBangBoard, Player, index_to_coord

def battle(gen1, gen2, num_games_each_side=10, search_num=200, noise=0.05):
    net_path = './data/nets'
    net1_filename = f"gen_{gen1}.net"
    net2_filename = f"gen_{gen2}.net"

    net1 = GoBangNet().cuda()
    net1.load_param(os.path.join(net_path, net1_filename))

    net2 = GoBangNet().cuda()
    net2.load_param(os.path.join(net_path, net2_filename))

    net1_win = 0
    net2_win = 0

    # 10 games with net1 as black
    for i in range(num_games_each_side):
        print("Game %d" % (i+1))

        tree1 = TreeSearch(net1)
        tree2 = TreeSearch(net2)

        game_now = GoBangBoard()
        player_now = Player.BLACK

        while not game_now.is_game_ended():
            if player_now == Player.BLACK:
                tree1.search(game_now)
                tree1.add_noise(game_now, noise)
                for search_count in range(search_num):
                    tree1.search(game_now)
                _, move = tree1.get_pi_and_get_move(0, game_now)
                game_now = game_now.move(move)
                player_now = Player.WHITE
            else:
                tree2.search(game_now)
                tree2.add_noise(game_now, noise)
                for search_count in range(search_num):
                    tree2.search(game_now)
                _, move = tree2.get_pi_and_get_move(0, game_now)
                game_now = game_now.move(move)
                player_now = Player.BLACK
            game_now.print_board()
            print('move: ', index_to_coord(move))

        if game_now.is_black_win():
            print("Net 1 wins")
            net1_win += 1
        else:
            print("Net 2 wins")
            net2_win += 1

        game_now.print_board()
        print('net 1 win count:', net1_win)
        print('net 2 win count:', net2_win)

    half_point = [net1_win, net2_win]
    net1_win = net2_win = 0

    # 10 games with net1 as white
    for i in range(num_games_each_side):
        print("Game %d" % (i+1))

        tree1 = TreeSearch(net2)
        tree2 = TreeSearch(net1)

        game_now = GoBangBoard()
        player_now = Player.BLACK

        while not game_now.is_game_ended():
            if player_now == Player.BLACK:
                tree1.search(game_now)
                tree1.add_noise(game_now, noise)
                for search_count in range(search_num):
                    tree1.search(game_now)
                _, move = tree1.get_pi_and_get_move(0, game_now)
                game_now = game_now.move(move)
                player_now = Player.WHITE
            else:
                tree2.search(game_now)
                tree2.add_noise(game_now, noise)
                for search_count in range(search_num):
                    tree2.search(game_now)
                _, move = tree2.get_pi_and_get_move(0, game_now)
                game_now = game_now.move(move)
                player_now = Player.BLACK
            game_now.print_board()
            print('move: ', index_to_coord(move))

        if game_now.is_black_win():
            print("Net 2 wins")
            net2_win += 1
        else:
            print("Net 1 wins")
            net1_win += 1
        game_now.print_board()

        print('net 1 win count:', net1_win)
        print('net 2 win count:', net2_win)
        print('half point', half_point)
    return half_point, [net1_win, net2_win]

# battle(6, 4, 3)