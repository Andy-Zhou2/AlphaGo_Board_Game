from TreeSearch import generate_single_game
from model import GoBangNet
import pickle
from multiprocessing import Pool, Process
from utils import *


def run_self_play(thread_id, gen, net):
    net = net.cuda()
    print('start working on', thread_id)
    single_game_data = generate_single_game(net, False, 800)

    with open(f'./data/games/1_game_800_gen_{gen}_thread_{thread_id}.pkl', 'wb') as f:
        pickle.dump(single_game_data, f)


def run_self_play_multi_thread(gen, game_num=40):
    net = GoBangNet()
    net.load_param(f'./data/nets/gen_{gen}.net')
    start_id = get_next_game_id(gen)
    with Pool(4) as p:
        p.starmap(run_self_play, [(i, gen, net) for i in range(start_id, start_id + game_num)])

if __name__ == '__main__':
    run_self_play_multi_thread(5)
