from dino_env import DinoEnv
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import load_model
import random
import pickle
import os
import time

BUFFER_SIZE = 60000

ACTIONS = 2  #  jump or keep stay
GAMMA = 0.99  # decay rate of past observations original 0.99
BATCH_SIZE = 16  # size of minibatch
LEARNING_RATE = 1e-4
EPSILON_DECAY = 0.99999 # decay rate of epsilon

OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon

EPSILON_FILE = "epsilon.pkl"
BUFFER_FILE = "buffer.pkl"
STEP_FILE = "step.pkl"
MODEL_FILE = "current.model"

img_rows, img_cols = 80, 80 #image size
img_channels = 4  # We stack 4 frames

def buildmodel(model_file = None):
    if model_file:
        model = load_model(model_file)
        print("load from existing model")
    else:
        print("build the model from 0")
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


def save_obj(obj, name):
    with open(name , 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name , 'rb') as f:
        return pickle.load(f)


def save_breakpoint(epison,step,model,file_name,data_buffer):
    model.save(file_name)
    save_obj(epison,EPSILON_FILE)
    save_obj(step, STEP_FILE)
    save_obj(data_buffer, BUFFER_FILE)


def train(env):
    mode_file = MODEL_FILE if os.path.exists(MODEL_FILE) else None
    model = buildmodel(mode_file)
    current_s = env.get_current_status()
    if os.path.exists(EPSILON_FILE):
        epsilon = load_obj(EPSILON_FILE)
    else:
        epsilon = 0.1
    if os.path.exists(STEP_FILE):
        step = load_obj(STEP_FILE)
    else:
        step = 0

    last_time = time.time()
    while True:
        Q_value = 0
        step += 1
        # epsilon *= EPSILON_DECAY # todo this method should be adjusted
        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and step > OBSERVATION:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if random.random() <= epsilon:  # exploration
            print("--------------random action ----------")
            # action = 0
            action = random.randrange(ACTIONS)
        else:  # exploitation
            q = model.predict(current_s)  # get the prediction by current status
            max_Q = np.argmax(q)  # choose action with maximum q value
            action = max_Q

        next_s, reward, isover, others = env.step(action)
        data_buffer.append((current_s, action, reward, next_s, isover))

        if isover:
            current_s = env.get_ini_status()
        else :
            current_s = next_s

        if (step) %100 == 0:
            print ("{} steps".format(step))

        if (step) % 1000 == 0:
            env.pause()
            save_breakpoint(epsilon,step,model,MODEL_FILE,data_buffer)
            env.resume()

        if len(data_buffer) > 200:
            # begin training
            minibatch = random.sample(data_buffer, BATCH_SIZE)
            status_0 = minibatch[0][0]
            inputs = np.zeros((BATCH_SIZE, status_0.shape[1], status_0.shape[2], status_0.shape[3]))  # 32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]  # stack of images
                action_t = minibatch[i][1]   
                reward_t = minibatch[i][2]  # reward at state_t and action_t
                state_t1 = minibatch[i][3]  # next state
                gameover = minibatch[i][4]  # if game over

                inputs[i:i + 1] = state_t
                # inputs[i] = state_t
                targets[i] = model.predict(state_t)  # predicted q values
                Q_value = model.predict(state_t1)  # predict q values for next step

                if gameover:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_value)  # with future reward
            # env.pause()
            # loss += model.train_on_batch(inputs, targets)
            loss = model.train_on_batch(inputs, targets)
            # env.resume()
            if step%10 == 0:
                print ("epsilon is {} and loss is {} and Q_value is {}".format(epsilon,loss,Q_value))
                print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate

        last_time = time.time()

env = DinoEnv()
#env = DinoEnv("./chromedriver")

data_buffer = load_obj(BUFFER_FILE) if os.path.exists(BUFFER_FILE) else deque(maxlen=BUFFER_SIZE)

try:
    train(env)
except Exception as e:
    print(e)
    env.close()


#todo batchsize  framerate random/action eisilon/declay/rate  reward/number

