from threading import Lock, Semaphore, Event, Thread
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from time import sleep
import logging
from logging import info, debug, error

PREDICTION_BATCH_SIZE = 2

queue_waiting_predictions = Queue()

queue_lock = Lock()
computing = Event()
not_computing = Event()
not_computing.set()

input_queue = []
results_queue = []


def prediction(boards):
    sleep(3)  # computation
    results_queue.append([f"result_{i}" for i in boards])

    not_computing.set()
    computing.clear()


def submit_for_prediction(board, count):
    global input_queue
    info(f"Submitting {count} for prediction")
    with queue_lock:
        not_computing.wait()
        info(f'processing {count}')
        input_queue.append(board)
        position = len(input_queue) - 1
        batch_number = len(results_queue)
        info(f'get info: {batch_number}, {position}, {count}, {input_queue}')
        if len(input_queue) == PREDICTION_BATCH_SIZE and not computing.is_set():
            Thread(target=prediction, args=[input_queue.copy()]).start()
            computing.set()
            not_computing.clear()
            input_queue = []
    computing.wait()
    not_computing.wait()
    info(f'get result for {count}: {results_queue[batch_number][position]}')


fmt = "%(asctime)s: %(message)s"
logging.basicConfig(format=fmt, level=logging.INFO,
                    datefmt="%H:%M:%S")
with ThreadPoolExecutor(max_workers=2) as executor:
    for index in range(20):
        executor.submit(submit_for_prediction, index, index)
print(results_queue)