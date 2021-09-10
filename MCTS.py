from board_tik_tak_toe import *

c_puct = 1


class Node:
    def __init__(self, s: Board, net):
        """创建节点，创建节点下的边，保存了N, s, v"""
        self.N = 0
        self.s = s
        self.edges = dict()  # a -> Edge
        p, v = net(s)
        self.v = v
        valid_actions = get_valid_actions(s)
        next_probabilities = normalize_actions_probability(p, valid_actions)
        for a in valid_actions:
            self.edges[a] = Edge(next_probabilities[a])

    def get_best_edge(self):
        max_u, best_a, best_edge = -float("inf"), [], Edge(0)
        for a in get_valid_actions(self.s):
            edge = self.edges[a]
            N = self.N
            u = edge.Q + c_puct * edge.P * N / (1 + edge.N)
            if u > max_u:
                max_u = u
                best_a = a
                best_edge = edge
        return max_u, best_a, best_edge


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

    max_u, best_a, best_edge = current_node.get_best_edge()

    if not best_edge.expanded:
        best_edge.expand(s.reverse(), net)  # 儿子节点的s应该是以白方视角的，因此reverse
        v = best_edge.son.v
    else:
        v = search(net, best_edge.son)
    v *= -1  # 之前的v其实是白色方的胜率，*-1后为黑色方胜率

    best_edge.Q = (best_edge.Q + v) / (best_edge.N + 1)
    best_edge.N += 1
    current_node.N += 1

    return v


class Tree:
    def __init__(self, net):
        self.root = Node(Board(None), net)  # None -> Empty Board
        self.net = net

    def search_from_root(self, search_time=1600):
        for _ in range(search_time):
            search(self.net, self.root)

    def next_round(self):
        """
        deletes the root node and other edges/nodes except the best one
        理论上垃圾回收机制能够回收掉
        但是可能还是手动删掉比较好
        :return:
        """
        last_root = self.root
        self.root = self.root.get_best_edge().son
        
