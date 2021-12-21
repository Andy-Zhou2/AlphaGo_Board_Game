from gobang_board import *
from random import randint
from math import sqrt

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
            # p = p.cpu()
            # v = v.cpu()
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
        print('counts:', counts)
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

    def get_pi_and_move(self, tau):
        pi_distribution = self.get_pi(self.root, tau)
        move = np.random.choice(225, p=pi_distribution)
        return pi_distribution, move

data = []
from model import GoBangNet
net = GoBangNet().cuda()
tree = TreeSearch(net)
while not tree.root.is_game_ended():
    t1 = time.time()
    tree.search_from_root(10)
    pi_distribution, move = tree.get_pi_and_move(0)
    tree.root = tree.root.move(move)
    data.append((tree.root, pi_distribution))
    tree.root.print_board()
    print(time.time() - t1)