from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event, Thread
from gobang_board import GoBangBoard
import torch as t
from tree_search_parallel_predictor import generate_single_game
from threading import Thread
from model import GoBangNet


PREDICTION_BATCH_SIZE = 1


class ParallelPredictor:
    def __init__(self, net):
        self.queue_lock = Lock()
        self.computing = Event()
        self.not_computing = Event()
        self.not_computing.set()

        self.input_queue = []
        self.results = []

        self.net = net

    def prediction(self, boards):
        n = t.Tensor(PREDICTION_BATCH_SIZE, 3, 15, 15).cuda()
        for i in range(PREDICTION_BATCH_SIZE):
            n[i] = boards[i]
        # boards = t.Tensor(boards).cuda()
        # print(f"Submitting {n.shape} for prediction")
        policy, value = self.net(n)
        self.results.append((policy.detach().cpu().numpy(), value.detach().cpu().numpy()))

        self.not_computing.set()
        self.computing.clear()

    def submit(self, board):
        with self.queue_lock:
            self.not_computing.wait()
            # info(f'processing {count}')
            self.input_queue.append(board)
            # print('input queue', len(self.input_queue))
            position = len(self.input_queue) - 1
            batch_number = len(self.results)
            # info(f'get info: {batch_number}, {position}, {input_queue}')
            if len(self.input_queue) == PREDICTION_BATCH_SIZE and not self.computing.is_set():
                Thread(target=self.prediction, args=[self.input_queue]).start()
                self.computing.set()
                self.not_computing.clear()
                self.input_queue = []
        self.computing.wait()
        self.not_computing.wait()
        # print(self.results[batch_number][0][position], self.results[batch_number][1][position])
        try:
            rt = self.results[batch_number][0][position], self.results[batch_number][1][position]
        except Exception as e:
            print(len(self.results))
            print(len(self.results[batch_number]))
            print(batch_number, position)
        return self.results[batch_number][0][position], self.results[batch_number][1][position]

    def predict(self, x):
        with t.no_grad():
            flag = False
            if isinstance(x, GoBangBoard):
                flag = True
                x = t.Tensor([x.black, x.white, x.turn])
                # x = x.unsqueeze(0)
                x = x.cuda()
            policy, value = self.submit(x)
            if flag:
                policy = policy[0]
                value = value[0]
            return policy, value



def run_self_play(gen_id):
    try:
        generate_single_game(net, gen_id, False, 200)
    except Exception as e:
        print(e)


from multiprocessing import Pool, Process

net = GoBangNet().cuda()
net.load_param('./data/nets/gen_0.net')

# predictor = ParallelPredictor(net)

if __name__ == '__main__':
    with Pool(3) as p:
        p.map(run_self_play, range(5))

    # p1 = Process(target=run_self_play, args=(0, predictor))
    # p1.start()
    # p1.join()

# generate_single_game(net, 0, False, 200)