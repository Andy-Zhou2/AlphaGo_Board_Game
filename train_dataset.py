import torch as t
from torch.utils.data import Dataset
import os
import pickle


def get_binary_info(path):
    with open(path, 'rb') as file:
        result = pickle.load(file)
    return result


data_path = rf'./data/games'
# filename = rf'test_data.pkl'
training_set = [] #get_binary_info(os.path.join(data_path, filename))

for i in range(1, 22):
    filename = f'5_games_200_gen_2_search_{i}.pkl'
    training_set.extend(get_binary_info(os.path.join(data_path, filename)))
for i in range(1, 22):
    filename = f'5_games_200_gen_2_script_1_search_{i}.pkl'
    training_set.extend(get_binary_info(os.path.join(data_path, filename)))


class GameData(Dataset):
    def __init__(self):
        super(GameData, self).__init__()

    def __len__(self):
        return len(training_set)

    def __getitem__(self, item_num):
        board, pi, z = training_set[item_num]

        sample = {"s": t.Tensor(board), "p": t.Tensor(pi), "z": t.Tensor([z])}
        return sample
