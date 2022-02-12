from gen_play import run_self_play_multi_thread
from train import train
from arena import battle
from datetime import datetime

battle(77, 77, search_num=1600, num_games_each_side=1)