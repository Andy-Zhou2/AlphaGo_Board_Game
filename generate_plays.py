from MCTS import Tree
from board_tik_tak_toe import *
import pickle
import os


class GameGenerator:
    def __init__(self, net, save_path):
        self.net = net
        self.save_path = save_path

    def generate_one_game(self, game_id: int):
        moves = []
        tree = Tree(self.net)
        while not tree.root.s.is_game_end():
            tree.search_from_root()
            moves.append(tree.next_round())  # next_round returns best_a
        with open(os.path.join(self.save_path, str(game_id)), 'wb') as f:
            pickle.dump(moves, f)

    def generate_games(self, num_games=2048):
        for _ in range(num_games):
            ...



