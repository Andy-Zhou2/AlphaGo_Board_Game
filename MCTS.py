from gobang_board import *
from random import randint
from math import sqrt

c_puct = 1


class Node:
    def __init__(self, s: GoBangBoard, net):
        """创建节点，创建节点下的边，保存了N, s, v"""
        self.N = 0
        self.s = s
        self.edges = dict()  # a -> Edge
        if s.is_game_ended():
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
        max_f, best_a, best_edge = -float("inf"), None, None
        for a in self.valid_actions:
            edge = self.edges[a]
            u = edge.Q + c_puct * edge.P * sqrt(self.N) / (1 + edge.N)
            if u > max_f:
                max_f = u
                best_a = a
                best_edge = edge
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
    if s.is_game_ended():
        print("game ended")
        print(s.print_board())
        print(s.reward())
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

    def search_from_root(self, search_time=1):
        for _ in range(search_time):
            # print('searching count: ', self.root.N)
            search(self.net, self.root)

    # def next_round(self, edge):
    #     """
    #     deletes the root node and other edges/nodes except the best one
    #     全靠垃圾回收机制
    #     :return:
    #     """
    #     # note: 最终选择的不是U最大的边，而是走过次数最多的边
    #     self.root = edge.son

    def select_edge_and_progress(self, tau) -> Edge:
        """
        selects edge and changes the root node to the son of best edge
        does not do searching!
        :param tau: temperature coefficient
        :return: edge
        """
        assert tau in (0, 1)
        if tau == 0:
            max_N, a, edge = self.root.get_best_N_edge()
            self.root = edge.son
        if tau == 1:
            a, edge = self.root.get_edge_prop_to_N()
            self.root = edge.son
        return edge


class SelfPlayGame:
    def __init__(self, net):
        """set up the board"""
        self.tree = Tree(net)
        self.progress = []
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
        self.tree.select_edge_and_progress(1 if self.move_count <= 30 else 0)
        self.progress.append(self.tree.root.s)
        print('move count: ', self.move_count)
        self.tree.root.s.print_board()

    def generate_whole_game(self):
        while not self.tree.root.s.is_game_ended():
            self.next_round()
        # TODO 倒推，判断每次的最终胜利者
        return self.progress


if __name__ == '__main__':
    from model import GoBangNet

    net = GoBangNet().cuda()
    l = []
    for i in range(100):
        g = SelfPlayGame(net)
        l.append(g.generate_whole_game())
        print(g.tree.root.s.print_board())
