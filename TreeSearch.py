from gobang_board import *
from random import randint
from math import sqrt
from model import GoBangNet
from gobang_board import get_symmetries
import numpy as np

c_puct = 1
DEFAULT_SEARCH_COUNT = 800


class TreeSearch:
    def __init__(self, net):
        self.net = net

        self.Qsa = dict()  # Q[(s, a)]
        self.Nsa = dict()  # N[(s, a)]
        self.Ns = dict()
        self.Ps = dict()  # P[s] is a vector, P[s][a]

        self.Es = dict()
        self.Rs = dict()
        self.Vs = dict()

        self.root = GoBangBoard()

    def search_every_move(self, s):
        board = s.get_str_representation()
        if board not in self.Es:
            self.Es[board] = s.is_game_ended()
        if self.Es[board]:
            if board not in self.Rs:
                self.Rs[board] = s.get_reward()
            return self.Rs[board]

        if board in self.Ns:  # must be, otherwise game is ending
            valid_moves = self.Vs[board]

            for a in range(225):
                if valid_moves[a]:
                    best_a = a
                    next_s = s.move(a)
                    v = self.search(next_s)

                    v *= -1  # changes into win rate of current player

                    self.Ns[board] += 1
                    if (board, best_a) in self.Qsa:
                        self.Qsa[(board, best_a)] = (self.Nsa[(board, best_a)] * self.Qsa[(board, best_a)] + v) / (
                                self.Nsa[(board, best_a)] + 1)
                        self.Nsa[(board, best_a)] += 1
                    else:
                        self.Qsa[(board, best_a)] = v
                        self.Nsa[(board, best_a)] = 1
        self.get_pi(s, 1)
        print('above is after every')

    def search(self, s):
        board = s.get_str_representation()
        if board not in self.Es:
            self.Es[board] = s.is_game_ended()
        if self.Es[board]:
            if board not in self.Rs:
                self.Rs[board] = s.get_reward()
            return self.Rs[board]

        if board in self.Ns:  # not a leaf node
            valid_moves = self.Vs[board]
            best_confidence = -float('inf')
            best_a = None

            for a in range(225):
                if valid_moves[a]:
                    if (board, a) in self.Qsa:
                        u = self.Qsa[(board, a)] + c_puct * self.Ps[board][a] * sqrt(self.Ns[board]) / (
                                1 + self.Nsa[(board, a)])
                    else:  # (s, a) not in Qsa and Nsa
                        u = c_puct * self.Ps[board][a] * sqrt(self.Ns[board] + 1e-7)
                        # Qsa[(s, a)] = 0, to prevent all zeros

                    # print(a, ':', u, file=open('file.txt', 'a'))
                    if u > best_confidence:
                        best_confidence = u
                        best_a = a
            # print('best u, a:', best_confidence, best_a, file=open('file.txt', 'a'))
            # print('----------------------', file=open('file.txt', 'a'))

            next_s = s.move(best_a)
            v = self.search(next_s)

            v *= -1  # changes into win rate of current player

            self.Ns[board] += 1
            if (board, best_a) in self.Qsa:
                self.Qsa[(board, best_a)] = (self.Nsa[(board, best_a)] * self.Qsa[(board, best_a)] + v) / (
                        self.Nsa[(board, best_a)] + 1)
                self.Nsa[(board, best_a)] += 1
            else:
                self.Qsa[(board, best_a)] = v
                self.Nsa[(board, best_a)] = 1

            return v
        else:  # is a leaf node
            v = self.expand(s)
            return v

    def expand(self, s):
        board = s.get_str_representation()
        p, v = self.net.predict(s)
        p = p.numpy()
        valid_moves = s.get_valid_actions()
        self.Vs[board] = valid_moves
        p = p * valid_moves
        p /= p.sum()
        self.Ps[board] = p
        self.Ns[board] = 0
        return v

    def get_pi(self, s, tau):
        board = s.get_str_representation()
        counts = np.array([self.Nsa[(board, a)] if (board, a) in self.Nsa else 0 for a in range(225)])

        args = np.flip(np.argsort(counts))
        print('top 10 moves')
        for i in range(225):  # print top 10 moves
            index = args[i]
            print(counts[index], index_to_coord(index),
                  self.Nsa[(board, index)] if (board, index) in self.Nsa else 0,
                  self.Qsa[(board, index)] if (board, index) in self.Qsa else "not in Qsa",
                  self.Ps[board][index],
                  self.Qsa[(board, index)] + c_puct * self.Ps[board][index] * sqrt(self.Ns[board]) / (
                          1 + self.Nsa[(board, index)]) if (board, index) in self.Qsa else
                  c_puct * self.Ps[board][index] * sqrt(self.Ns[board] + 1e-7)
                  )

        # counts = counts * self.Vs[s]
        if np.max(counts) == 0:
            print('warning! max N is 0!')
        if tau == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            result = [0] * len(counts)
            result[bestA] = 1
            return result
        else:
            counts = counts ** (1 / tau)
            result = counts / counts.sum()
            return result

    def search_from_root(self, search_count=DEFAULT_SEARCH_COUNT):
        for i in range(search_count):
            self.search(self.root)
            print('search count:', i)

    def get_pi_and_get_move(self, tau, target=None):
        if target is None:
            target = self.root
        pi_distribution = self.get_pi(target, tau)
        move = np.random.choice(225, p=pi_distribution)
        return pi_distribution, move

    def progress(self, move):
        self.root = self.root.move(move)
        # board = self.root.get_str_representation()
        self.add_noise(self.root, 0.3)
        # self.Ps[board] = 0.1 * np.random.dirichlet(np.ones(225)) + self.Ps[board]

    def add_noise(self, s, noise_level=0.3):
        board = s.get_str_representation()
        if board not in self.Ps:
            print('note! add noise to un-expanded node')
            self.expand(s)
        self.Ps[board] = noise_level * np.random.dirichlet(np.ones(225)) + self.Ps[board]


def generate_single_game(net, print_every_step=False, sim_per_step=200):
    black = np.zeros([15, 15])
    white = np.zeros([15, 15])
    turn = np.zeros([15, 15])
    black[0, 0:4] = 1
    init_board = GoBangBoard((black, white, turn))


    t1 = time.time()
    data = []
    tree = TreeSearch(net)

    tree.root = init_board
    tree.expand(tree.root)
    tree.add_noise(tree.root, 0.3)

    move_count = 0
    while not tree.root.is_game_ended():
        t2 = time.time()
        tree.search_every_move(tree.root)
        tree.search_from_root(sim_per_step)
        pi_distribution, move = tree.get_pi_and_get_move(tau=0 if move_count > 15 else 1)
        new_data = get_symmetries((tree.root.black, tree.root.white, tree.root.turn), pi_distribution)
        data.append(new_data)
        if tree.Nsa[(tree.root.get_str_representation(), move)] == 0:
            print('note: selecting a move with 0 visit')
            board = tree.root.get_str_representation()
            print(np.array([tree.Nsa[(board, a)] if (board, a) in tree.Nsa else 0 for a in range(225)]))
            print(move)
        tree.progress(move)
        move_count += 1
        if print_every_step:
            tree.root.print_board()
        print('time per step:', time.time() - t2)
        import sys
        sys.exit(0)

    print(tree.root.get_reward())
    tree.root.print_board()
    print(tree.root.current_player)

    r = tree.root.get_reward()
    r *= -1
    all_data = []
    for i in range(len(data) - 1, -1, -1):
        for sym in range(8):
            data[i][sym].append(r)
        r *= -1
        all_data.extend(data[i])

    print('generate single game time (s):', time.time() - t1)
    return data
