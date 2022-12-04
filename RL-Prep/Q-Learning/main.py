import numpy as np
import gym
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
'''
Observation:
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right#
'''

lr = 0.1
gamma = 0.95
n_games = 60000
prev_score = 0
score_history = []
average_scores = []

# epsilon is associated with how random you take an action.
epsilon = 1

#exploration is decaying and we will get to a state of full exploitation
epsilon_decay_value = 0.99995

def get_discrete_state(state):
    step_size = np.array([0.1, 0.1, 0.1, 0.1])
    discrete_state = state/step_size + np.array([1,1,1,1])
    return tuple(discrete_state.astype('int'))

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.tight_layout()
    plt.show()
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    
    #randomly initializing values in our q table our q table
    q_table_shape = [45, 45, 45, 45]
    q_table = np.random.uniform(low=0, high=1, size=(q_table_shape + [env.action_space.n]))

    if not os.path.exists("./logs"):
        os.makedirs("./logs/plots")
        os.makedirs("./logs/tabulated_records")
        os.makedirs("./logs/checkpoints")
    
    plot_name = 'ql-cartpole_log_0-1_0-95.png'
    figure_file = './logs/plots/' + plot_name

    for i in range(n_games):
        observation = env.reset()
        discrete_state = get_discrete_state(observation)
        done = False
        score = 0 

        while not done: 
            #if some random number is greater than epsilon, then we take the best possible action we have explored so far
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])

            #else, we will explore and take a random action
            else:
                action = np.random.randint(0, env.action_space.n)

            observation_, reward, done, info = env.step(action)
            score += reward

            # Need new state for Q-learning algorithm
            new_discrete_state = get_discrete_state(observation_)

            if not done:
                max_new_q = np.max(q_table[new_discrete_state])

                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - lr) * current_q + lr * (reward + gamma* max_new_q)

                q_table[discrete_state + (action,)] = new_q
            
            # set the state just chosen as the "original" state
            discrete_state = new_discrete_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        average_scores.append(float(avg_score))

        # print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        if epsilon > 0.05: 
            if score > prev_score and n_games > int(n_games/6):
                epsilon = math.pow(epsilon_decay_value, n_games - int(n_games/6))
        prev_score = score

    ac_scores = pd.DataFrame(
        {
            "q-learning" : average_scores
        }
    )
    x = [i+1 for i in range(n_games)]
    ac_scores.to_csv("./logs/tabulated_records/ql-log.csv", index=False)
    plot_learning_curve(x, score_history, figure_file)