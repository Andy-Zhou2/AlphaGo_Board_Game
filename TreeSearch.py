from gobang_board import *
from random import randint
from math import sqrt
from model import GoBangNet
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
                        u = c_puct * self.Ps[board][a] * sqrt(self.Ns[board])  # Qsa[(s, a)] = 0

                    if u > best_confidence:
                        best_confidence = u
                        best_a = a

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
            p, v = self.net.predict(s)
            p = p.numpy()
            # v = v.cpu()
            valid_moves = s.get_valid_actions()
            # print('type v:', type(valid_moves))
            self.Vs[board] = valid_moves
            p = p * valid_moves
            p /= p.sum()
            self.Ps[board] = p
            self.Ns[board] = 0
            # print('type:', type(p))
            return v

    def get_pi(self, s, tau):
        board = s.get_str_representation()
        counts = np.array([self.Nsa[(board, a)] if (board, a) in self.Nsa else 0 for a in range(225)])
        # print('counts:', counts)
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

    def get_pi_and_get_move(self, tau, target=None):
        if target is None:
            target = self.root
        pi_distribution = self.get_pi(target, tau)
        move = np.random.choice(225, p=pi_distribution)
        return pi_distribution, move

    def progress(self, move):
        self.root = self.root.move(move)
        board = self.root.get_str_representation()
        # print(type(self.Ps[board]))
        self.Ps[board] = 0.1 * np.random.dirichlet(np.ones(225)) + self.Ps[board]


def generate_single_game(net, print_every_step=False):
    t1 = time.time()
    data = []
    tree = TreeSearch(net)
    while not tree.root.is_game_ended():
        tree.search_from_root(200)
        pi_distribution, move = tree.get_pi_and_get_move(tau=0)
        data.append([(tree.root.black, tree.root.white, tree.root.turn), pi_distribution])
        # tree.root = tree.root.move(move)
        tree.progress(move)
        if print_every_step:
            tree.root.print_board()

    print(tree.root.get_reward())
    tree.root.print_board()
    print(tree.root.current_player)

    r = tree.root.get_reward()
    r *= -1
    for i in range(len(data) - 1, -1, -1):
        data[i].append(r)
        r *= -1

    print('generate single game time (s):', time.time() - t1)
    # for i in data:
    #     print(i)
    return data

# generate_single_game()
