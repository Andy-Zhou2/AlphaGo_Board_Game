import torch as t
from torch.utils.data import Dataset
import os
import pickle


def get_binary_info(path):
    with open(path, 'rb') as file:
        result = pickle.load(file)
    return result


data_path = rf'./data/games'
filename = rf'test_data.pkl'
training_set = get_binary_info(os.path.join(data_path, filename))


class GameData(Dataset):
    def __init__(self):
        super(GameData, self).__init__()

    def __len__(self):
        return len(training_set)

    def __getitem__(self, item_num):
        board, pi, z = training_set[item_num]

        sample = {"s": t.Tensor(board), "p": t.Tensor(pi), "z": t.Tensor([z])}
        return sample
