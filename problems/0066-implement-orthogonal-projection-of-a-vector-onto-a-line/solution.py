
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	
	if len(v) != len(L):
		return -1

	inner_product = 0
	for i in range(len(v)):
		inner_product += v[i] * L[i]
	
	norm = 0
	for i in range(len(L)):
		norm += L[i] * L[i]

	if norm != 0:
		coef = inner_product / norm
	else:
		coef = 0

	project = [u * coef for u in L]
	project = [round(x,3) for x in project]

	return  project
