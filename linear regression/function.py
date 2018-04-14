"""
@brief Returns the equation of the line passing through points p1 and p2.

"""
def linear_function(p1, p2):
	m = (p2[1] - p1[1])/(p2[0] - p1[0])
	b = p1[1] - m*p1[0]
	
	return lambda x: m*x + b

def quadratic_function(x):
	return x[1]**2 + x[2]**2 - 0.6