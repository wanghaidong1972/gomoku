# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from policy_value_net_keras import PolicyValueNet # Keras

import multiprocessing
from multiprocessing.pool import Pool
import time

import copy
import os.path
import logging
logging.basicConfig(filename='training.log',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')


def single_game_play():
    board = Board(width=15,height=15,n_in_row=5)
    game = Game(board)
    temp = 1.0
    player = MCTS_Pure()
    winner, play_data = game.start_self_play(player, temp=temp)
    return winner, play_data

class TrainPipeline():
    def __init__(self, init_model=None):
        #cpu count
        self.n_workers = multiprocessing.cpu_count() -1
        self.worker_pool = None

        #
        self.episode_len = -1

        # params of the board and the game
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 36000
        self.batch_size = 1024  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.pre_data_size = 5
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        # self.check_freq = 50
        self.check_freq = 30 # when use predfined data
        self.game_batch_num = 300
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.batch_i = 0

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            self.best_win_ratio = 0.6
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
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

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            print("batch i:{}, episode_len:{}".format(self.batch_i + 1, self.episode_len))
            logging.info("batch i:{}, episode_len:{}".format(self.batch_i + 1, self.episode_len))
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def collect_pre_data(self, n_games=10):
        """collect self-play data for training"""
        contents = self.game.load_predata(game_count=n_games)

        i = 0
        for content in contents:
            i = i+1
            try:
                winner, play_data = content
            except Exception as e:
                print(e)
                continue # just ignore this line

            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            print("load data i:{}, episode_len:{}".format(i, self.episode_len))
            logging.info("load data i:{}, episode_len:{}".format(i + 1, self.episode_len))
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        loginfo = ("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new)
        print(loginfo)
        logging.info(loginfo)
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.batch_i = i
                self.collect_pre_data(self.pre_data_size)
                if i > 1000 : # now the model should be applied to self play .
                    self.collect_selfplay_data(self.play_batch_size) # use self play as well
                    self.check_freq = 10 # increase check frequency when use self played data too.

                # print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                print("batch i:{} is training".format(i+1))
                logging.info("batch i:{} is training".format(i + 1))
                # logging.info("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    loginfo = "current self-play batch: {}".format(i+1)
                    print(loginfo)
                    logging.info(loginfo)
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        logging.info("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    if os.path.exists('./current_policy.model'):
        training_pipeline = TrainPipeline(init_model='./current_policy.model')
        print('use existing model file')
        logging.info('use existing model file')
    else:
        print('train from 0')
        logging.info('train from 0')
        training_pipeline = TrainPipeline()
    training_pipeline.run()

# todo

# load trained model to continue
# save record of auto-play(at least when vs pure mcts ) ->sgf format
# simple gui to load record and show -> parse sgf