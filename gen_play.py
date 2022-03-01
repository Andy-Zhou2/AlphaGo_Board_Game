from TreeSearch import generate_single_game
from model import GoBangNet
import pickle
from multiprocessing import Pool, Process
from utils import *


def run_self_play_once(gen):
    net = GoBangNet()
    net.load_param(f'./data/nets/gen_{gen}.net')
    net = net.cuda()
    single_game_data = generate_single_game(net, True, 800)

    with open(f'./data/games/1_game_800_gen_{gen}.pkl', 'wb') as f:
        pickle.dump(single_game_data, f)

if __name__ == '__main__':
    run_self_play_once(0)
