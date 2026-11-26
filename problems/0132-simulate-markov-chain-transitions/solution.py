import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):

    current_state = initial_state
    num_state = np.arange(transition_matrix.shape[0])
    trajetory = []
    trajetory.append(current_state)
    for i in range(num_steps):
        current_state = np.random.choice(num_state, p=transition_matrix[current_state])
        trajetory.append(current_state)
    
    return trajetory