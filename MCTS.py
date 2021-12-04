from board_tik_tak_toe import *
from random import randint

c_puct = 1


class Node:
    def __init__(self, s: Board, net):
        """创建节点，创建节点下的边，保存了N, s, v"""
        self.N = 0
        self.s = s
        self.edges = dict()  # a -> Edge
        p, v = net(s)
        self.v = v
        self.valid_actions = get_valid_actions(s)
        next_probabilities = normalize_actions_probability(p, self.valid_actions)
        for a in self.valid_actions:
            self.edges[a] = Edge(next_probabilities[a])

    def get_best_U_edge(self):
        max_u, best_a, best_edge = -float("inf"), [], Edge(0)
        for a in self.valid_actions:
            edge = self.edges[a]
            N = self.N
            u = edge.Q + c_puct * edge.P * N / (1 + edge.N)
            if u > max_u:
                max_u = u
                best_a = a
                best_edge = edge
        return max_u, best_a, best_edge

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
        edges = []
        total_N = 0
        for a in self.edges:
            edge = self.edges[a]
            edges.append(edge)
            total_N += edge.N
        print('total N for the node', total_N)

        select = randint(1, total_N)

        prev_N = 0
        for a in self.edges:
            edge = self.edges[a]
            prev_N += edge.N
            if prev_N >= select:
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
    无论棋盘具体情况都是假设当前是该黑色走了
    返回当前黑色的胜率
    """
    s = current_node.s
    if s.is_game_end():
        return reward(s)

    max_u, best_a, best_edge = current_node.get_best_U_edge()

    if not best_edge.expanded:
        best_edge.expand(next_state(s, best_a).reverse(), net)  # 儿子节点的s应该是以白方视角的，因此reverse
        v = best_edge.son.v
    else:
        v = search(net, best_edge.son)
    v *= -1  # 之前的v其实是白色方的胜率，*-1后为黑色方胜率

    best_edge.Q = (best_edge.Q + v) / (best_edge.N + 1)
    best_edge.N += 1
    current_node.s.print_self()
    current_node.N += 1
    print(current_node)

    return v


class Tree:
    def __init__(self, net):
        self.root = Node(Board(None), net)  # None -> Empty Board
        self.net = net

    def search_from_root(self, search_time=15):
        for _ in range(search_time):
            search(self.net, self.root)
            print(self.root.N)

    def next_round(self, edge):
        """
        deletes the root node and other edges/nodes except the best one
        全靠垃圾回收机制
        :return:
        """
        # note: 最终选择的不是U最大的边，而是走过次数最多的边
        self.root = edge.son

    def select_edge(self, tau):
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
        self.move = 0

    def next_round(self):
        """
        1. search 1600 rounds
        2. call next_round() of Tree
        3. record progress
        :return:
        """
        self.move += 1
        self.tree.search_from_root()
        print(self.tree.root.N)
        edge = self.tree.select_edge(1 if self.move <= 30 else 0)
        self.progress.append(edge)
        self.tree.next_round(edge)

    def generate_whole_game(self):
        while not self.tree.root.s.is_game_end():
            self.next_round()
        # TODO 倒推，判断每次的最终胜利者
        return self.progress


if __name__ == '__main__':
    from model import GoBangNet
    net = GoBangNet()
    g = SelfPlayGame(net)
    print(g.generate_whole_game())
