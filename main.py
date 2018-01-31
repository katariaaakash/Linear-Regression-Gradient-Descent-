from numpy import *

def compute_error_for_given_points(b, m, points):
	total_error = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		m = points[i, 1]
		total_error += (y - (m*x + b)) **2
	return total_error / float(len(points))


def step_gradient(b_currunt, m_currunt, points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - (m_currunt*x) + b_currunt)
		m_gradient += -(2/N) * x * (y - (m_currunt*x) + b_currunt)
	new_b = b_currunt - (learning_rate * b_gradient)
	new_m = m_currunt - (learning_rate * m_gradient)
	return [new_b, new_m]



def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
	b = initial_b
	m = initial_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b,m]

def run():
	points = genfromtxt('data.csv', delimiter=',')
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print(b)
	print(m)



if __name__ == '__main__':
	run()