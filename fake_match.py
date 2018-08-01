from game import *
from random_player import *
from mcts_pure import *


def run():
    n = 4
    width, height = 6, 6

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        player2 = Random_plyer()
        # player2 = Random_plyer()
        player1 = MCTSPlayer()

        # set start_player=0 for human first
        game.start_play(player1, player2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()