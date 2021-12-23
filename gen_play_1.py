from TreeSearch import generate_single_game
from model import GoBangNet
import pickle

net = GoBangNet().cuda()
net.load_param('./data/nets/gen_0.net')

count = 1
while True:
    data = []
    for i in range(5):
        data.extend(generate_single_game(net, True, 800))

    with open(f'./data/games/5_games_800_gen_0_script_1_search_{count}.pkl', 'wb') as f:
        pickle.dump(data, f)
    count += 1
