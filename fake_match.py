from game import *
from random_player import *
from mcts_pure import *
from mp_mcts import *

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    n = 5
    width, height = 8,8

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # player2 = Random_player()
        player2 = MPlayer(n_playout=20)
        # player1 = Human()
        player1 = MCTSPlayer()

        # set start_player=0 for human first
        game.start_play(player1, player2, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()