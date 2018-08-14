import multiprocessing
import time
from random import randint

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net_keras import PolicyValueNet # Keras
from mcts_alphaZero import MCTSPlayer

def wrapper(args):
    return single_game_play(*args)

def single_game_play(num,initmode):
    print('Starting worker {} and initmode is {}'.format(num,initmode))
    board = Board(width=8,height=8,n_in_row=5)
    game = Game(board)
    temp = 1.0
    # player = MCTS_Pure()
    if initmode:
        # start training from an initial policy-value net
        policy_value_net = PolicyValueNet(8,8,model_file=initmode)
    else:
        policy_value_net = PolicyValueNet(8, 8)

    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400,
                                 is_selfplay=1)

    winner, play_data = game.start_self_play(mcts_player,temp=temp)
    print('Exiting worker{}'.format(num))
    return winner, play_data


PROCESSES = 3
WORKER_CALLS = 3

def worker(num):
    """worker function"""
    print ('Starting worker {}'.format(num))
    time.sleep(randint(2,4))
    print ('Exiting worker{}'.format(num))
    return "ok"

if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=PROCESSES)
    # pool_outputs = pool.map(worker, range(WORKER_CALLS))
    param_list = [(i,"./best_policy.model") for i in range(WORKER_CALLS)]
    pool_outputs = pool.map(wrapper,param_list)
    pool.close()
    pool.join()
    print ('Pool finished')


    # winner,result = single_game_play(1)
    # print ("winner is {} ".format(winner))
