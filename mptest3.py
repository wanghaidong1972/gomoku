import multiprocessing
from multiprocessing.pool import Pool
import time

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure

def single_game_play():
    board = Board(width=8,height=8,n_in_row=5)
    game = Game(board)
    temp = 1.0
    player = MCTS_Pure()
    winner, play_data = game.start_self_play(player, temp=temp)
    return winner, play_data

def dump_func(procnum):
    print ("in process ".format(procnum))
    for i in range(10):
        print("process {} to sleep {} seconds ".format(procnum,i))
        # time.sleep(i)
    return procnum, 999

if __name__ == '__main__':
    n_workers = multiprocessing.cpu_count() -1
    print("we have {}cores".format(n_workers+1))

    worker_pool = Pool(processes=n_workers)
    incoming = []  # queue for result to process
    ongoing = []  # queue for running job
    data_buffer = []

    i = 0
    while i < 3:
        if len(ongoing) >= n_workers:
            # Too many playouts running? will not happen in our simple case
            # ongoing[0][0].wait(0.01 / n_workers)
            pass
        else:
            i += 1
            # ongoing.append(worker_pool.apply_async(single_game_play))
            ongoing.append(worker_pool.apply_async(dump_func,i))

        while incoming:
            winner, play_data = incoming.pop()
            play_data = list(play_data)[:]
            data_buffer.extend(play_data)

        for job in ongoing:
            if not job.ready():
                continue
            winner, play_data = job.get()
            incoming.append((winner, play_data))
            ongoing.remove(job)

    worker_pool.close()
    worker_pool.join()
    print("result" + " ".join(data_buffer))
