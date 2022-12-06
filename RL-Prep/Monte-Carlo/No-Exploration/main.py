import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#Ignore gym messages below that of error severity
gym.logger.set_level(40)

#Build environment
env = gym.make("Blackjack-v1")

#Build spaces
#agent sum space
sum_space = [i for i in range(env.observation_space[0].n)]
#dealer card space
dealer_space = [i for i in range(env.observation_space[1].n)]
#player ace space
ace_space = [False, True]
#action space
action_space = [i for i in range(env.action_space.n)]

#Initialise parameters
Q = {} # Q-table
state_space = [] # A collection of the different combinations of the above spaces
returns = {} 
pairs_visited = {} # States and action pairs visited - records the number of times each state and concurrent action is visited

# Above dictionaries (not the state space list) are populated with combos of each item of the above spaces
# each entry is a tuple(tuple(total, card, ace), action) - first entry of tuple matches observation space, second is just the action
for total in sum_space:
    for card in dealer_space:
        for ace in ace_space:
            for action in action_space:
                Q[((total, card, ace), action)] = 0
                returns[((total, card, ace), action)] = 0
                pairs_visited[((total, card, ace), action)] = 0
            # state space list populated with combos of above spaces without reference to action in the form of tuple(total, card, ace)
            state_space.append((total, card, ace))
epsillon = 0.05
gamma = 1.0

#Build policies for any given state - this is where we use the state_space list
#Each policy will start as a completely random choice from the action space (in this case 0 or 1)
policy = {}
for state in state_space:
    policy[state] = np.random.choice(action_space)

n_games = 1000000
for i in tqdm(range(n_games)):
    #print(f'Game: {i}')
    states_actions_returns = [] # empty list to store states actions and returns
    memory = [] # Empty list to store memory of episode
    
    observation = env.reset()
    done = False

    while not done:
        # Our observation details our current state
        # So we use that observation to select the action from our policy dictionary
        action = policy[observation]

        # Take step with that action - returns new observation plus others
        observation_, reward, done, info = env.step(action)

        # Append details of previous observation (the one used to select the policy action)
        # As well as the chosen action and reward
        memory.append((observation[0], observation[1], observation[2], action, reward))

        # Overwrite past observation with new observation to repeat while loop
        observation = observation_

    # When while loop ends append details of final observation
    # Loop breaks on done = True, therefore will break before observation is overwritten or memory is added to
    memory.append((observation[0], observation[1], observation[2], action, reward))

    # Ignore the last one - but we keep it in so we can create a real value for G on the first iteration of the loop
    G = 0
    last = True
    for player_total, dealer_card, useable_ace, action, reward in reversed(memory): # reverse to do last entry in the memory first
        if last:
            last = False
        else:
            states_actions_returns.append((player_total, dealer_card, useable_ace, action, G))
        G = gamma * G + reward # G gets larger as we move closer to start of loop - i.e. start of the memory - start of the game

    states_actions_returns.reverse() # reverse to put it in chronological order
    states_actions_visited = []

    # Loop over states_actions_returns list for each set of parameters
    for player_total, dealer_card, useable_ace, action, G in states_actions_returns:
        # Create state action pair
        sa = ((player_total, dealer_card, useable_ace), action)
        
        # Have we visited this state action pair before?
        # if not...
        if sa not in states_actions_visited:
            # add 1 to the count of the particular within pairs_visited dictionary
            pairs_visited[sa] += 1

            # using incremental implementation of the update rule for the agent's estiamte of discounter future rewards
            # shortcut that saves calculating the average of a function every single time
            # - Computationally expensive and provides little in terms of accuracy
            # new estimate = 1 / N * (sample - old estimate)

            returns[(sa)] += (1 / pairs_visited[(sa)]) * (G - returns[(sa)])
            Q[sa] = returns[sa]
            rand = np.random.random()
            if rand < 1 - epsillon:
                state = (player_total, dealer_card, useable_ace)
                values = np.array([Q[(state, a)] for a in action_space])
                best = np.random.choice(np.where(values==values.max())[0])
                policy[state] = action_space[best]
            else:
                policy[state] = np.random.choice(action_space)
            states_actions_visited.append(sa)
    if epsillon - 1e-7 > 0:
        epsillon -= 1e-7
    else:
        epsillon = 0
    
# Test
numEpisodes = 1000
rewards = np.zeros(numEpisodes)
totalReward = 0
wins = 0
losses = 0
draws = 0
print('getting ready to test policy')   
for i in range(numEpisodes):
    observation = env.reset()
    done = False
    while not done:
        action = policy[observation]
        observation_, reward, done, info = env.step(action)            
        observation = observation_
    totalReward += reward
    rewards[i] = totalReward

    if reward >= 1:
        wins += 1
    elif reward == 0:
        draws += 1
    elif reward == -1:
        losses += 1
    
wins /= numEpisodes
losses /= numEpisodes
draws /= numEpisodes
print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
plt.plot(rewards)
plt.show()    

