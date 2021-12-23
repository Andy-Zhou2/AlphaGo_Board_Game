from TreeSearch import generate_single_game
from model import GoBangNet
import pickle

net = GoBangNet().cuda()
net.load_param('./data/nets/test_1_model_DNN.net')

count = 1
while True:
    data = []
    for i in range(5):
        data.extend(generate_single_game(net, True))

    with open(f'./data/games/5_games_200_gen_2_search_{count}.pkl', 'wb') as f:
        pickle.dump(data, f)
    count += 1
