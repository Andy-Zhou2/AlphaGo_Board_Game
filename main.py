import os

from gen_play import run_self_play_once
from train import train
from arena import battle
from datetime import datetime
from utils import get_next_game_id

start_gen = 7417

def log(*msg):
    print(datetime.now(), *msg, file=open('log.txt', 'a'))

if __name__ == '__main__':
    gen = start_gen
    log(f'start main with {start_gen}')

    while True:
        log(f'start gen {gen}')
        run_self_play_once(gen)
        log(f'{gen} finished generating self-play')
        log(train(gen, num_epoch=1, lr=0.002))

        if gen % 40 != 0:
            os.remove(f'./data/nets/gen_{gen}.net')

        log(f'{gen} has finished')
        gen += 1
