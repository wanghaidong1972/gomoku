# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: wang haidong
"""

import numpy as np


class Random_player(object):

    def __init__(self,player=1):
        self.player = player

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        pass

    def get_action(self, board):
        sensible_moves  = board.availables
        if len(sensible_moves) > 0:
            move = np.random.choice(sensible_moves)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Random {}".format(self.player)

if __name__ == '__main__':
    board = np.random.randint(100, size=10)
    player1 = Random_player()
    print(player1.get_action(board))