import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	l2_norm = 0
	for i in range(len(gradient)):
		l2_norm += gradient[i]**2
	l2_norm = np.sqrt(l2_norm)

	if l2_norm != 0:
		u_a = [gradient[i] / l2_norm  for i in range(len(gradient))]
		u_d = [-1 * gradient[i] / l2_norm  for i in range(len(gradient))]
	else:
		u_a = gradient
		u_d = gradient
		
	return {
		'magnitude': l2_norm,
		'direction': u_a,
		'descent_direction': u_d
	}