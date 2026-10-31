from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""

	if h == 0:
		raise ValueError('ValueError')

	n = len(point)

	def second_order_derivate(f, point, h, i, j):
		a = [value for value in point]
		a[i-1] += h 
		a[j-1] += h
		b = [value for value in point]
		b[i-1] -= h 
		b[j-1] += h 
		c = [value for value in point]
		c[i-1] += h 
		c[j-1] -= h
		d = [value for value in point]
		d[i-1] -= h
		d[j-1] -= h

		f_2nd_ij  = ((f(a)-f(b))-(f(c)-f(d))) / (4*h*h)

		return f_2nd_ij
	
	Hess = [[0]*n for _ in range(n)]

	for i in range(n):
		for j in range(n):
			Hess[i][j] = second_order_derivate(f,point,h,i+1,j+1)
	
	return Hess

		