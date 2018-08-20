# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
""" 

from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

from keras.utils import np_utils

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4  # coef of l2 penalty
        if model_file:
            #   net_params = pickle.load(open(model_file, 'rb'))
            #   self.model.set_weights(net_params)
            self.model = load_model(model_file)
        else:
            self.create_policy_value_net()
        self._loss_train_op()

        # self.policy_value = self.policy_value()
        
    def create_policy_value_net(self):
        """create the policy value network """   
        in_x = network = Input((4, self.board_width, self.board_height))

        # conv layers
        '''
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        '''

        layer1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        layer2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(layer1)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)

        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(in_x, [self.policy_net, self.value_net])

        '''
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value
        '''

    def policy_value(self,state_input):
        state_input_union = np.array(state_input)
        results = self.model.predict_on_batch(state_input_union)
        return results

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        # net_params = self.get_policy_param()
        # pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
        # self.model.save_weights(model_file)
        self.model.save(model_file)

    def _conv_bn_relu(self,nb_filter, nb_row, nb_col, subsample=(1, 1)):
        def f(input):
            conv = self.Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                                 init="he_normal", border_mode="same")(input)
            norm = BatchNormalization(mode=0, axis=1)(conv)
            return Activation("relu")(norm)

        return f

    def _bn_relu_conv(self,nb_filter, nb_row, nb_col, subsample=(1, 1)):
        def f(input):
            norm = BatchNormalization(mode=0, axis=1)(input)
            activation = Activation("relu")(norm)
            return self.Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                                 init="he_normal", border_mode="same")(activation)

        return f

    def _basic_block(self,nb_filters, init_subsample=(1, 1)):
        def f(input):
            conv1 = self._bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
            residual = self._bn_relu_conv(nb_filters, 3, 3)(conv1)
            return self._shortcut(input, residual)

        return f

    def _shortcut(self,input, residual):
        stride_width = input._keras_shape[2] / residual._keras_shape[2]
        stride_height = input._keras_shape[3] / residual._keras_shape[3]
        equal_channels = residual._keras_shape[1] == input._keras_shape[1]

        shortcut = input
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = self.Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                     subsample=(stride_width, stride_height),
                                     init="he_normal", border_mode="valid")(input)

        return merge([shortcut, residual], mode="sum")

    def _residual_block(self,block_function, nb_filters, repetations, is_first_layer=False):
        def f(input):
            for i in range(repetations):
                init_subsample = (1, 1)
                if i == 0 and not is_first_layer:
                    init_subsample = (2, 2)
                input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
            return input

        return f

    def resnet(self):
        input = Input(shape=(3, 224, 224))

        conv1 = self._conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = self.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        # Build residual blocks..
        block_fn = self._basic_block
        block1 = self._residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
        block2 = self._residual_block(block_fn, nb_filters=128, repetations=4)(block1)
        block3 = self._residual_block(block_fn, nb_filters=256, repetations=6)(block2)
        block4 = self._residual_block(block_fn, nb_filters=512, repetations=3)(block3)

        # Classifier block
        pool2 = self.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=1000, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model
