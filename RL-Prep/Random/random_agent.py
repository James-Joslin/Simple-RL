import gym
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.tight_layout()
    plt.plot(x, running_avg)
    plt.savefig(figure_file)

if __name__ == '__main__':
    # env = gym.make('ALE/AirRaid-v5')
    env = gym.make("CartPole-v1")
    n_games = 1000

    if not os.path.exists("./logs"):
        os.makedirs("./logs/plots")
        os.makedirs("./logs/tabulated_records")
        os.makedirs("./logs/checkpoints")

    plot_name = 'cartpole_log.png'
    figure_file = './logs/plots/' + plot_name

    best_score = env.reward_range[0]
    score_history = []
    average_scores = []
    load_checkpoint = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            # print(action, action.shape)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        average_scores.append(float(avg_score))

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        ac_scores = pd.DataFrame(
            {
                "Actor-Critic" : average_scores
            }
        )
        ac_scores.to_csv("./logs/tabulated_records/random-log.csv", index=False)
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)