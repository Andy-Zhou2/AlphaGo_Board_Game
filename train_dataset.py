import torch as t
from torch.utils.data import Dataset
import os
import pickle
from utils import *


def get_binary_info(path):
    with open(path, 'rb') as file:
        result = pickle.load(file)
    return result

class GameData(Dataset):
    def __init__(self, gen):
        super(GameData, self).__init__()
        data_path = rf'./data/games'

        filename = f'1_game_800_gen_{gen}.pkl'
        print(filename, 'has #data', len(get_binary_info(os.path.join(data_path, filename))))

        self.training_set = get_binary_info(os.path.join(data_path, filename))

        print('training set length', len(self.training_set))

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, item_num):
        board, pi, z = self.training_set[item_num]

        sample = {"s": t.Tensor(board), "p": t.Tensor(pi), "z": t.Tensor([z])}
        return sample
