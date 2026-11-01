def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	expectation = (n+1) / 2
	var = 0
	for i in range(n):
		var += (i+1-expectation)**2
	var /= n
	
	return expectation, var