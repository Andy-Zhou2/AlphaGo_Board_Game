from MCTS import SelfPlayGame
from board_tik_tak_toe import *
import pickle
import os


class GameGenerator:
    def __init__(self, net, save_path):
        self.net = net
        self.save_path = save_path

    def generate_one_game(self):
        """
        注意要转换成训练用的数据
        :return:
        """
        game = SelfPlayGame(self.net)
        return game.generate_whole_game()

    def generate_games(self, num_games=2048):
        for _ in range(num_games):




