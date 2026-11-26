import numpy as np

def simulate_mdp(P, R, policy, start_state, terminal_states, gamma, num_episodes, max_steps, seed=42):
    """
    Simulate episodes in an MDP and compute discounted returns.
    
    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states)
        R: Rewards, shape (n_states, n_actions, n_states)
        policy: Stochastic policy, shape (n_states, n_actions)
        start_state: Initial state for each episode
        terminal_states: List of terminal state indices
        gamma: Discount factor
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (episode_returns, average_return)
    """
    
    np.random.seed(seed)
    episodes_returns = []
    n_states = np.arange(P.shape[0])
    n_actions = np.arange(P.shape[1])

    for j in range(num_episodes):
        per_episode_return = 0
        state = start_state
        action = 0
        for i in range(max_steps):
            if state in terminal_states:
                break
            else:
                action = np.random.choice(n_actions, p=policy[state,:])
                next_state = np.random.choice(n_states, p=P[state, action, :])
                weighted_reward = R[state, action, next_state] * gamma**i

                per_episode_return += weighted_reward
                state = next_state
        episodes_returns.append(per_episode_return)

    average_returns = 0
    for reward in episodes_returns:
        average_returns += reward
    average_returns = average_returns / len(episodes_returns)

    return episodes_returns, round(average_returns,4)


    
    