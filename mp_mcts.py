# -*- coding: utf-8 -*-
"""
A multi-processes implementation of the Monte Carlo Tree Search (MCTS)

@author: wang haidong
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.last_leafvalue = 0

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

        if self.is_root():
            self.last_leafvalue = leaf_value

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_children(self):
        return self._children

    def get_visits(self):
        return self._n_visits

    def set_parent(self,parent):
        self._parent = parent

class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000, root=None):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        if root is not None:
            self._root = root
            self._root._parent = None
            self._realroot = False  # this flag is for multiprocessing
        else:
            self._root = TreeNode(None, 1.0)
            self._realroot = True

        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def callbac_updateroot(self):
        pass
        #update root value by children's value

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root

        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_root_node(self):
        return self._root


    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        # this method should only be called when self is real root.,so that's here where we can should use mutiprocess
        if self._root.is_leaf(): # no expanded children yet
            action_probs, _ = self._policy(state)
            self._root.expand(action_probs)

        for n in range(self._n_playout):
            # get top n (assumed to be 6) nodes from children
            if n%6 == 0:
                the_children = self._root._children
                top_n = sorted(the_children.items(),key=lambda act_node: act_node[1].get_value(self._c_puct))[:6]
                for child_node in top_n:
                    # child_tree = MCTS(policy_value_fn,copy.deepcopy(child_node)) # use copy because we will use it in multiprocess
                    child_tree = MCTS(policy_value_fn,
                                      child_node)  # use copy because we will use it in multiprocess
                    state_copy = copy.deepcopy(state)
                    state_copy.do_move(child_node[0])
                    child_tree._playout(state_copy)
                    self._root.update(-child_tree.get_root_node().last_leafvalue) # update real root
                    child_tree.get_root_node().set_parent(self._root) # to link the sub tree
                    # self._root.get_children()[child_node[0]] = child_tree.get_root_node() # copy sub tree

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1].get_visits())[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=20):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            if move is None:
                print (sensible_moves)
            self.mcts.update_with_move(-1)
            # self.reset_player()
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


# todo
# the search tree is reset every time it return a move now  ,but we should  reuse it for the next 2 step.
# deal with the edge case of virtua loss such as like “what if the same leaf gets selected twice despite the virtual loss”
# and “tree consists of one root node”
# or to avoid those troublesome edge cases , just start root paralleling after we have some children