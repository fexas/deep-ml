import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	
	n = len(P)

	entropy  = 0
	cross_entropy = 0

	for i in range(n):
		if P[i] > 0:
			entropy -= P[i] * np.log(P[i])
			if Q[i] > 0:
				cross_entropy -= P[i] * np.log(Q[i])
			elif Q[i] == 0: 
				cross_entropy = float('inf')
			else:
				raise ValueError('ValueError')
		elif P[i] ==0:
			entropy -= 0
		else:
			raise ValueError('ValueError')

	return entropy, cross_entropy