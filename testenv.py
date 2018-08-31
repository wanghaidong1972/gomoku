from  dino_env import DinoEnv
import numpy as np
from collections import deque

# keras imports
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import random


buffer_size = 1200
data_buffer = deque(maxlen=buffer_size)

ACTIONS = 2  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 16  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    return model

model = buildmodel()
env = DinoEnv("./chromedriver")
current_s = env.get_current_status()

steps = np.random.randint(0,2,500)

loss = 0
Q_sa = 0
for i,action in enumerate(steps):

    stack, reward,isover,others = env.step(action)
    data_buffer.append((current_s, action, reward, stack, isover))

    if isover:
        current_s = env.get_ini_status()
    else :
        current_s = stack


    if i%100 == 0:
        print ("{} steps".format(i))

        print ("stack len is {} and stack shape is {}  reward is {}  over? is {}".format(
            len(stack),stack[0].shape,reward,isover
        ))

    if len(data_buffer) > 200:
        # begin train
        # sample a minibatch to train on
        minibatch = random.sample(data_buffer, BATCH)
        s_t = minibatch[0][0]
        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
        targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

        # Now we do the experience replay
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]  # 4D stack of images
            action_t = minibatch[i][1]  # This is action index
            reward_t = minibatch[i][2]  # reward at state_t due to action_t
            state_t1 = minibatch[i][3]  # next state
            terminal = minibatch[i][4]  # wheather the agent died or survided due the action

            # inputs[i:i + 1] = state_t
            inputs[i] = state_t
            targets[i] = model.predict(state_t)  # predicted q values
            Q_sa = model.predict(state_t1)  # predict q values for next step

            if terminal:
                targets[i, action_t] = reward_t  # if terminated, only equals reward
            else:
                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
        # game_state._game.pause()
        loss += model.train_on_batch(inputs, targets)
        # game_state._game.resume()
        print ("loss is {} and Q_sa is {}".format(loss,Q_sa))




