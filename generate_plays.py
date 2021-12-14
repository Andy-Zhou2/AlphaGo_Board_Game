from MCTS import SelfPlayGame
from model import GoBangNet
import time
from gobang_board import *
import pickle
import os


class GameGenerator:
    def __init__(self, save_path, net=None):
        self.net = GoBangNet().cuda() if net is None else net
        self.net.eval()
        self.save_path = save_path

    def generate_games(self, num_games=2048):
        for game_count in range(num_games):
            start_time = time.time()
            print(f'generating game no {game_count}')
            game = SelfPlayGame(self.net)
            single_game_data = game.generate_whole_game()
            end_time = time.time()
            print(f'game no {game_count} generated in {end_time - start_time} seconds')
            with open(os.path.join(self.save_path, f'game_{game_count}.pkl'), 'wb') as f:
                pickle.dump(single_game_data, f)


g = GameGenerator(save_path='./data/games/')
g.generate_games()
