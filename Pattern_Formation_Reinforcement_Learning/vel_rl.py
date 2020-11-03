from drones import Drones
import time

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import adam

def read_points(e):
    file_index = int(e/100)
    file_path = "/home/saran/Desktop/projects/rl/drones/Input_data/inp_" + str(file_index) + ".txt"

    starting_points = list()
    final_points = list()
    n = 0

    with open(file_path) as e:
        n = int(e.readline())
        for i in range(n):
            line = e.readline()
            tempx, tempy = line.split(" ")
            x = int(tempx)
            y = int(tempy)
            point = dict({"x": x, "y":y})
            starting_points.append(point)
        for i in range(n):
            line = e.readline()
            tempx, tempy = line.split(" ")
            x = int(tempx)
            y = int(tempy)
            point = dict({"x": x, "y":y})
            final_points.append(point)
    
    return n, starting_points, final_points

n, starting_points, final_points = read_points(0)
env = Drones(n, starting_points, final_points)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 35
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_set = list()
            for i in range(n):
                action_set.append(random.randrange(3))
            return np.array(action_set)
        act_values = self.model.predict(state)
        act_values = np.reshape(act_values, (n,3))
        # print(act_values)
        return np.argmax(act_values, axis=1)
    
    def get_actions_hash(self, actions):
        actions_hash = list()
        actions_list = actions.tolist()
        for i in range(len(actions_list)):
            actions_hash.append(np.sum(actions_list[i]))
        return np.array(actions_hash)


    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # print(actions)

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        actions_hash = self.get_actions_hash(actions) 

        # print(actions_hash)
        # print(actions)

        targets_full[[ind], [actions_hash]] = targets
        # print(targets_full)

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episode, n, starting_points, final_points):

    loss = []
    agent = DQN(n*3, n*3)
    print(agent.model.summary)
    for e in range(episode):

        #read data points for every 100th episode
        if (e % 100 == 0) and (e != 0):
            n, starting_points, final_points = read_points(e)
        
        state = env.reset(n, starting_points, final_points)
        state = np.reshape(state, (1, 3*n))
        score = 0
        max_steps = 1000
        print(e)
        for i in range(max_steps):
            action = agent.act(state)
            # print(action)
            # print()
            reward, next_state, done = env.step(action, n)
            time.sleep(0.1)
            score += reward
            next_state = np.reshape(next_state, (1, 3*n))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
    
    print(agent.model.summary)
    agent.model.save("Drones_RL_Velocity_Model.h5")

    return loss

if __name__ == '__main__':

    ep = 100
    loss = train_dqn(ep, n, starting_points, final_points)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()