import time

from gobang_board import *
from random import randint
from math import sqrt

c_puct = 1
DEFAULT_SEARCH_COUNT = 200

Es = dict()  # stores if game is over
Vs = dict()  # stores valid actions for s


def get_game_ended(s):
    if s.get_str_representation() in Es:
        return Es[s.get_str_representation()]
    else:
        result = s.is_game_ended()
        Es[s.get_str_representation()] = result
        return result




class Node:
    def __init__(self, s: GoBangBoard, net):
        """创建节点，创建节点下的边，保存了N, s, v"""
        self.N = 0
        self.s = s
        self.edges = dict()  # a -> Edge
        if get_game_ended(s):
            self.v = s.reward()
            self.valid_actions = []
            return
        p, v = net(s)
        self.v = v
        self.valid_actions = s.get_valid_actions()
        next_probabilities = normalize_actions_probability(p, self.valid_actions)
        for a in self.valid_actions:
            self.edges[a] = Edge(next_probabilities[a])

    def get_next_search_edge(self):
        # t1 = time.time()
        max_f, best_a, best_edge = -float("inf"), None, None
        for a in self.valid_actions:
            edge = self.edges[a]
            u = edge.Q + c_puct * edge.P * sqrt(self.N) / (1 + edge.N)
            if u > max_f:
                max_f = u
                best_a = a
                best_edge = edge
        # print("get_next_search_edge: ", time.time() - t1)
        return max_f, best_a, best_edge

    def get_best_N_edge(self):
        max_N, best_a, best_edge = -float("inf"), [], Edge(0)
        for a in self.valid_actions:
            edge = self.edges[a]
            N = self.N
            if N > max_N:
                max_N = N
                best_a = a
                best_edge = edge
        return max_N, best_a, best_edge

    def get_edge_prop_to_N(self):
        rand_num = randint(1, self.N)

        prev_N = 0
        for a in self.edges:
            edge = self.edges[a]
            prev_N += edge.N
            if prev_N >= rand_num:
                return a, edge

    def get_pi_distribution(self, tau):
        """
        assume that the node is the root of the tree
        returns probability proportional to N (if tau = 1) or only the target node (if tau = 0)
        :return:
        """
        result = [0] * 15 * 15
        assert tau == 0 or tau == 1
        if tau == 1:
            total_N = 0
            for i in range(15 * 15):
                a = index_to_coord(i)
                if a in self.valid_actions:
                    edge_N = self.edges[a].N
                    result[i] = edge_N
                    total_N += edge_N
            for i in range(15 * 15):
                result[i] /= total_N
        else:  # tau == 0
            max_N, best_a, best_edge = self.get_best_N_edge()
            result[coord_to_index(best_a)] = 1
        return result


class Edge:
    def __init__(self, p_a):
        self.W = 0
        self.P = p_a
        self.Q = 0
        self.N = 0
        self.expanded = False
        self.son = None

    def expand(self, s, net):
        self.expanded = True
        self.son = Node(s, net)


def search(net, current_node):
    """
    搜索单位是node，不是edge
    返回当前玩家的胜率
    """
    # print("searching...")
    # current_node.s.print_board()
    s = current_node.s
    if get_game_ended(s):
        return s.reward()

    # otherwise, select best edge and search
    max_s, best_a, best_edge = current_node.get_next_search_edge()

    if not best_edge.expanded:
        # print('expanding: ', best_a)
        # s.move(*best_a).print_board()
        best_edge.expand(s.move(*best_a), net)
        v = best_edge.son.v
    else:
        v = search(net, best_edge.son)
    # v is the win rate of next player
    v *= -1  # changes into win rate of current player

    # update edge
    best_edge.W += v
    best_edge.N += 1
    best_edge.Q = best_edge.W / best_edge.N
    # update node
    current_node.N += 1

    return v


class Tree:
    def __init__(self, net):
        self.root = Node(GoBangBoard(), net)  # create empty board
        self.net = net

    def search_from_root(self, search_time=DEFAULT_SEARCH_COUNT):
        for _ in range(search_time):
            search(self.net, self.root)

    def select_edge_and_progress(self, tau) -> list:
        """
        selects edge and changes the root node to the son of best edge
        does not do searching! assume that already searched
        returns the pi distribution
        :param tau: temperature coefficient
        :return: pi distribution
        """
        assert tau in (0, 1)
        pi_distribution = self.root.get_pi_distribution(tau)
        if tau == 0:
            max_N, a, edge = self.root.get_best_N_edge()
            self.root = edge.son
        if tau == 1:
            a, edge = self.root.get_edge_prop_to_N()
            self.root = edge.son
        return pi_distribution


class SelfPlayGame:
    def __init__(self, net):
        """set up the board"""
        self.tree = Tree(net)
        self.s = []
        self.pi = []
        self.move_count = 0

    def next_round(self):
        """
        1. search 1600 rounds
        2. call next_round() of Tree
        3. record progress
        :return:
        """
        self.move_count += 1
        self.tree.search_from_root()
        self.s.append(self.tree.root.s)
        # select tree, also saves the distribution
        self.pi.append(self.tree.select_edge_and_progress(1 if self.move_count <= 30 else 0))

        print('move count: ', self.move_count)
        # self.tree.root.s.print_board()

    def generate_whole_game(self):
        """
        generate the whole game, returns a list of data (s, pi, z)
        :return:
        """
        while not self.tree.root.s.is_game_ended():
            t1 = time.time()
            self.next_round()
            print('time: ', time.time() - t1)

        training_data_single_game = []
        winner = Player.BLACK if self.move_count % 2 == 1 else Player.WHITE
        print('winner: ', winner)
        self.tree.root.s.print_board()
        for step in range(self.move_count):
            x = self.s[step]
            unit_data = [[x.black, x.white, x.turn], self.pi[step], 1 if x.current_player == winner else -1]
            training_data_single_game.append(unit_data)

        return training_data_single_game

# if __name__ == '__main__':
#     from model import GoBangNet
#
#     net = GoBangNet().cuda()
#     l = []
#     # for i in range(100):
#     g = SelfPlayGame(net)
#     l.append(g.generate_whole_game())
#     # print(l)
#     # print(l[0][1])
