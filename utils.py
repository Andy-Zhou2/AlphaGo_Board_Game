import os

games_path = './data/games'

def get_next_game_id(gen):
    n = 0
    while True:
        if os.path.exists(os.path.join(games_path, f'1_game_800_gen_{gen}_thread_{n}.pkl')):
            n += 1
        else:
            return n


