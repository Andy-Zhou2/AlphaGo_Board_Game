import logging
import threading
import time
import concurrent.futures
from model import GoBangNet
import os
import torch as t

save_path = './data/nets'
if 'baseline.net' not in os.listdir(save_path):
    new_net = GoBangNet()
    t.save({'weight': new_net.state_dict()}, save_path + '/baseline.net')


