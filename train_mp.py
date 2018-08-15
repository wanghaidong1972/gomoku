# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNet # Keras

import multiprocessing
from multiprocessing.pool import Pool
import time

import copy
import os.path
import logging
logging.basicConfig(filename='training_mp.log',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

board_width = 6
board_height = 6
n_in_row = 4
learn_rate = 2e-3
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
temp = 1.0  # the temperature param
n_playout = 400  # num of simulations for each move
c_puct = 5
buffer_size = 12000
batch_size = 512  # mini-batch size for training
# batch_size = 2048  # mini-batch size for training
data_buffer = deque(maxlen=buffer_size)
play_batch_size = 9
epochs = 5  # num of train_steps for each update
kl_targ = 0.02
check_freq = 50
game_batch_num = 50
best_win_ratio = 0.0
# num of simulations used for the pure mcts, which is used as  the opponent to evaluate the trained policy
pure_mcts_playout_num = 1000
episode_len = -1
played_game = 0

def wrapper(args):
    return single_game_play(*args)

def single_game_play(num,initmode):
    print('Starting worker {} '.format(num))
    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row)
    game = Game(board)
    if initmode:
        policy_value_net = PolicyValueNet(board_width,board_height,model_file=initmode)
    else:
        policy_value_net = PolicyValueNet(board_width,board_height)

    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=c_puct,
                                 n_playout=n_playout,
                                 is_selfplay=1)

    winner, play_data = game.start_self_play(mcts_player,temp=temp)
    #should not do following line because zip function return a iterator instead of a static data strutcure like list
    #playlen = len(list(play_data))
    #print('Exiting worker{} and len is {}'.format(num,playlen))
    #logging.info('Exiting worker{} and len is {}'.format(num,playlen))
    return winner, play_data

def get_equi_data( play_data):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_porb.reshape(board_height, board_width)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data

def collect_selfplay_data( n_games=10,mp=False):
    global played_game
    global pool
    global episode_len
    """collect self-play data for training"""
    param_list = [(i, initmode) for i in range(n_games)]
    if mp:
        n_process = multiprocessing.cpu_count() -1
        pool = multiprocessing.Pool(processes=n_process )
        pool_outputs = pool.map(wrapper, param_list)
        pool.close()
        pool.join()

        for pool_output in pool_outputs:
            play_data = list(pool_output[1])[:]
            episode_len = len(play_data)
            # augment the data
            play_data = get_equi_data(play_data)
            data_buffer.extend(play_data)

        played_game = played_game + n_games
    else:
        played_game = played_game + 1
        winner, play_data = wrapper(param_list[0])
        play_data = list(play_data)[:]
        datalen = len(play_data)
        print("batch i:{}, episode_len:{}".format(played_game,datalen))
        logging.info("batch i:{}, episode_len:{}".format(played_game,datalen))
        # augment the data
        play_data = get_equi_data(play_data)
        data_buffer.extend(play_data)

def policy_update():
    global lr_multiplier
    """update the policy-value net"""
    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                learn_rate*lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
        )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         np.var(np.array(winner_batch)))
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         np.var(np.array(winner_batch)))
    loginfo = ("kl:{:.5f},lr_multiplier:{:.3f},loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new)
    print(loginfo)
    logging.info(loginfo)
    return loss, entropy

def policy_evaluate( n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                     c_puct=c_puct,
                                     n_playout=n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5,
                                 n_playout=pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row)
    game = Game(board)

    for i in range(n_games):
        winner = game.start_play(current_mcts_player,
                                      pure_mcts_player,
                                      start_player=i % 2,
                                      is_shown=0)
        win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(pure_mcts_playout_num,win_cnt[1], win_cnt[2], win_cnt[-1]))
    logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(pure_mcts_playout_num,win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio

def do_run():
    global win_ratio,best_win_ratio,pure_mcts_playout_num

    try:
        for i in range(game_batch_num):
            collect_selfplay_data(play_batch_size,mp=True)
            # print("batch i:{}, episode_len:{}".format(i+1, episode_len))
            print("played games{}".format(played_game))
            logging.info("played games{}".format(played_game))
            print("len of buffer is {}".format(len(data_buffer)))
            if len(data_buffer) > batch_size:
                loss, entropy = policy_update()
            # check the performance of the current model and save the model params
            if (i+1) % check_freq == 0:
                loginfo = "current self-play batch: {}".format(i+1)
                print(loginfo)
                logging.info(loginfo)
                win_ratio = policy_evaluate()
                policy_value_net.save_model('./current_policy.model')
                if win_ratio > best_win_ratio:
                    print("New best policy!!!!!!!!")
                    logging.info("New best policy!!!!!!!!")
                    best_win_ratio = win_ratio
                    # update the best_policy
                    policy_value_net.save_model('./best_policy.model')
                    if (best_win_ratio == 1.0 and
                            pure_mcts_playout_num < 5000):
                        pure_mcts_playout_num += 1000
                        best_win_ratio = 0.0
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    if os.path.exists('./current_policy.model'):
        initmode = './current_policy.model'
        policy_value_net = PolicyValueNet(board_width, board_height, model_file=initmode)
        logging.info('use existing model file')
        win_ratio = 0.6
    else:
        initmode = None
        policy_value_net = PolicyValueNet(board_width,board_height)
        win_ratio = 0.1

    do_run()


# todo

# load trained model to continue
# save record of auto-play(at least when vs pure mcts ) ->sgf format
# simple gui to load record and show -> parse sgf