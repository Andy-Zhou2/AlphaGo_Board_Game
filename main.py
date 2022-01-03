from gen_play import run_self_play_multi_thread
from train import train
from arena import battle
from datetime import datetime

start_gen = 9

def log(*msg):
    print(datetime.now(), *msg, file=open('log.txt', 'a'))

if __name__ == '__main__':
    log(f'start main with {start_gen}')
    while True:
        log(train(start_gen))
        [[w1, l1], [w2, l2]] = battle(start_gen + 1, start_gen)
        log(w1, l1, w2, l2)
        if w1 + w2 > l1 + l2:
            break
    log(f'{start_gen} has finished')
    start_gen += 1

    while True:
        while True:
            run_self_play_multi_thread(start_gen, 40)
            log(f'{start_gen} finished generating self-play')
            log(train(start_gen))
            [[w1, l1], [w2, l2]] = battle(start_gen+1, start_gen)
            log(w1, l1, w2, l2)
            if w1 + w2 > l1 + l2:
                break
        log(f'{start_gen} has finished')
        start_gen += 1
