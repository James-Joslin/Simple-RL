import gym
import numpy as np
from actor_critic import Agent
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
    agent = Agent(alpha=5e-4, n_actions=env.action_space.n)
    n_games = 500

    if not os.path.exists("./logs"):
        os.makedirs("./logs/plots")
        os.makedirs("./logs/tabulated_records")
        os.makedirs("./logs/checkpoints")

    plot_name = 'ac-cartpole_log_512_256_0-0005_0-99.png'
    figure_file = './logs/plots/' + plot_name

    best_score = env.reward_range[0]
    score_history = []
    average_scores = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)[0]
            # print(action, action.shape)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        average_scores.append(float(avg_score))

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        ac_scores = pd.DataFrame(
            {
                "Actor-Critic" : average_scores
            }
        )
        ac_scores.to_csv("./logs/tabulated_records/actor-critic-log.csv", index=False)
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)