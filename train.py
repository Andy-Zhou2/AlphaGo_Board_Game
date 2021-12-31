import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_dataset import GameData
from model import GoBangNet
import os
import time


def loss_function(policy_out, value_out, policy_pi, value_z):
    l = ((value_z - value_out) ** 2).squeeze(1) - t.sum(policy_pi * t.log(policy_out + 1e-7), dim=1)
    l = t.sum(l, dim=0)
    return l


def train(old_gen):
    GPU = t.device("cuda:0")
    net = GoBangNet()
    net.load_param(f'./data/nets/gen_{old_gen}.net')
    net = net.to(GPU)

    train_dataset = GameData(old_gen)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=0)

    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(1, 41):
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
        if epoch == 1:
            first_loss = loss_epoch
        if epoch == 40:
            last_loss = loss_epoch
        epoch_time = time.time() - epoch_start_time
        print('epoch time:', epoch_time)

    state = {"weight": net.state_dict()}

    t.save(state, os.path.join('./data/nets', f'gen_{old_gen+1}.net'))
    return first_loss, last_loss