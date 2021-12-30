from TreeSearch import generate_single_game
from model import GoBangNet
import pickle
from multiprocessing import Pool, Process

net = GoBangNet().cuda()
net.load_param('./data/nets/gen_4.net')

def run_self_play(thread_id):
    print('start working on', thread_id)
    single_game_data = generate_single_game(net, False, 800)

    with open(f'./data/games/1_game_800_gen_3_thread_{thread_id}.pkl', 'wb') as f:
        pickle.dump(single_game_data, f)


if __name__ == '__main__':
    with Pool(4) as p:
        p.map(run_self_play, range(40))
