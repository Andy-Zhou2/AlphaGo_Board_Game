import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_dataset import GameData
from model import GoBangNet
import os
import time


def loss_function(policy_out, value_out, policy_pi, value_z):
    # print(policy_out.device)
    # print(value_out.device)
    # print(policy_pi.device)
    # print(value_z.device)
    # policy_out += 1e-7
    # print(t.min(policy_out))
    l = ((value_z - value_out) ** 2).squeeze(1) - t.sum(policy_pi * t.log(policy_out + 1e-7), dim=1)
    # print(((value_z - value_out) ** 2).squeeze(1).shape)
    # print((t.sum(policy_pi * t.log(policy_out), dim=1).shape))
    # print(l.shape)
    l = t.sum(l, dim=0)
    # print('loss:', l)
    return l


if __name__ == '__main__':
    GPU = t.device("cuda:0")
    net = GoBangNet()
    net.load_param('./data/nets/test_1_model_DNN.net')
    net = net.to(GPU)

    train_dataset = GameData()
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=0)

    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)

    loss_all = []
    last_change_lr = 0
    model_count = 1
    for epoch in range(1, 40):
        print(f'start_epoch: {epoch}')
        epoch_start_time = time.time()
        loss_epoch = 0

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()  # gradient reset
            s, p, z = data["s"].to(GPU), data["p"].to(GPU), data["z"].to(GPU)

            policy_out, value_out = net(s)
            loss = loss_function(policy_out, value_out, p, z)

            loss_epoch += loss.item()
            loss.backward()

            optimizer.step()

        print('epoch, loss:', epoch, loss_epoch)
        epoch_time = time.time() - epoch_start_time
        print('epoch time:', epoch_time)

    state = {"weight": net.state_dict()}
    # if 'model' not in os.listdir('./data/nets'):
    #     os.mkdir('model')
    t.save(state, os.path.join('./data/nets', f'test_2_model_DNN.net'))
