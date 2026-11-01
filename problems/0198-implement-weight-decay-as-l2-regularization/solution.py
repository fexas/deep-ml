def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	# Your code here
	
	groups = len(parameters)

	for i in range(groups):
		d = len(parameters[i])
		if apply_to_all[i] == True:
			for j in range(d):
				parameters[i][j] = (1 - lr * weight_decay)*parameters[i][j] - lr * gradients[i][j]
		else:
			for j in range(d):
				parameters[i][j] -= lr * gradients[i][j]
			
	return parameters