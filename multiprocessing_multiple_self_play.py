from multiprocessing import Pool

from TreeSearch import TreeSearch
from model import GoBangNet
import pickle
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from model import GoBangNet
from gobang_board import get_symmetries
import numpy as np
import time

net = GoBangNet().cuda()
net.load_param('./data/nets/gen_0.net')


def generate_single_game(net, index, print_every_step=False, sim_per_step=200):
    print(f'start working on {index}')
    data = []
    tree = TreeSearch(net)
    move_count = 0
    while not tree.root.is_game_ended():
        t1 = time.time()
        tree.search_from_root(sim_per_step)
        pi_distribution, move = tree.get_pi_and_get_move(tau=0 if move_count > 15 else 1)
        new_data = get_symmetries((tree.root.black, tree.root.white, tree.root.turn), pi_distribution)
        data.append(new_data)
        if tree.Nsa[(tree.root.get_str_representation(), move)] == 0:
            print('note: selecting a move with 0 visit')
            board = tree.root.get_str_representation()
            print(np.array([tree.Nsa[(board, a)] if (board, a) in tree.Nsa else 0 for a in range(225)]))
            print(move)
        tree.progress(move)
        move_count += 1
        print(f'move time for {index} (s):', time.time() - t1)
        if print_every_step:
            tree.root.print_board()

    print(tree.root.get_reward())
    tree.root.print_board()
    print(tree.root.current_player)

    r = tree.root.get_reward()
    r *= -1
    all_data = []
    for i in range(len(data) - 1, -1, -1):
        for sym in range(8):
            data[i][sym].append(r)
        r *= -1
        all_data.extend(data[i])

    return data


def run_self_play(gen_id):
    try:
        generate_single_game(net, gen_id, False, 200)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    with Pool(3) as p:
        p.map(run_self_play, range(5))

# with ThreadPoolExecutor(max_workers=5) as executor:
#     for index in range(5):
#         executor.submit(run_self_play, index)
#
# run_self_play(0)

# count = 1
# while True:
#     data = []
#     for i in range(5):
#         data.extend(generate_single_game(net, True, 800))
#
#     with open(f'./data/games/5_games_800_gen_0_script_1_search_{count}.pkl', 'wb') as f:
#         pickle.dump(data, f)
#     count += 1
