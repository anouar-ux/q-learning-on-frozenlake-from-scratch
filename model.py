"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    q_table=np.zeros((num_states,num_actions))
    return q_table
    pass

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    return max(q_table[state])
    pass

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state]))
    pass

# Step 4 - sample_random_action
def sample_random_action(action_space):
    random=action_space.sample()
    return int(random)
    pass

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    random=rng.random()
    if random<epsilon:
        return True
    else:
        return False
    pass

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)

    Bool= should_explore(epsilon,rng)
    
    if Bool:
        return sample_random_action(action_space)
    else:
        M=max(q_table[state])
        index=[]
        for k in range(len(q_table[state])):
            if np.isclose(q_table[state][k],M):
                index.append(k)
        return int(rng.choice(index))

    pass

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon * decay_rate)
    pass

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.

    if done:
        return reward
    else:
        return reward+gamma*max_q_value(q_table,next_state)
    pass

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)

    temp_diff= target - q_table[state][action]

    return temp_diff
    pass

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    
    target= td_target(reward, gamma, q_table, next_state, done)
    error= td_error(target, q_table, state, action)

    q_table[state][action]+= alpha*(error)

    return q_table[state][action]
    
    pass

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    
    action_space=env.action_space
    action=epsilon_greedy_action(q_table, state, epsilon, action_space, rng)

    next_state, reward, terminated, truncated, info =env.step(action)

    done= bool(terminated) or bool(truncated)

    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)

    return (int(next_state), float(reward), done)
    pass

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    
    state,info=env.reset()
    steps=0
    final_reward=0

    while steps<max_steps:
        steps+=1
        next_state, reward, done =interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        final_reward+=reward
        if done:
            break
        state=next_state
    
    return final_reward
    pass

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    # TODO: train a Q-learning agent for num_episodes; return (q_table, returns)
    
    
    num_states, num_actions=env.observation_space.n,env.action_space.n
    q_table=init_q_table(num_states, num_actions)
    episode_returns=[]
    
    rng=np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.reset(seed=seed)

    epsilon=epsilon_start

    for _ in range(num_episodes):
        reward=run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps)
        epsilon=decay_epsilon(epsilon, epsilon_decay, epsilon_min)
        episode_returns.append(reward)


    return(q_table, episode_returns)
    pass

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.
    
    n_states=len(q_table)
    greedy_policy=np.array([0 for _ in range(n_states)], dtype=np.int64)

    for k in range(n_states):

        greedy_policy[k]=np.argmax(q_table[k])
    
    return greedy_policy

    pass

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # TODO: reset env, follow policy[state] each step, return bool(success)

    state, info=env.reset(seed=seed)
    steps=0
    done=False
    reward=0
    while steps<max_steps:
        steps+=1
        action=policy[state]
        next_state, reward, terminated, truncated, info =env.step(action)
        done=terminated or truncated
        if done:
            break
        state= next_state
    if reward==1:
        return True
    else:
        return False
    pass

# Step 16 - evaluate_success_rate
def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    # TODO: run num_episodes greedy rollouts and return the fraction that reached the goal.
    fraction=0
    for k in range(num_episodes):
        episode_seed=seed+k
        env.action_space.seed(episode_seed)

        Bool=run_greedy_episode(env, policy, episode_seed, max_steps)
        if Bool:
            fraction+=1

    return fraction/num_episodes

    pass

