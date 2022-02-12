from gen_play import run_self_play_multi_thread
from train import train
from arena import battle
from datetime import datetime
from utils import get_next_game_id

start_gen = 78

def log(*msg):
    print(datetime.now(), *msg, file=open('log.txt', 'a'))

if __name__ == '__main__':
    log(f'start main with {start_gen}')

    while True:
        while True:
            run_self_play_multi_thread(start_gen, 40)
            log(f'{start_gen} finished generating self-play')
            log(train(start_gen, num_epoch=1, lr=1e-4))
            [[w1, l1], [w2, l2]] = battle(start_gen+1, start_gen, num_games_each_side=15)
            log(w1, l1, w2, l2)
            if w1 > l1 and w2 > l2 and w1+w2 > l1+l2+3: # win by a margin
                break
        log(f'{start_gen} has finished')
        start_gen += 1
