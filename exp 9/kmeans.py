from math import *
import random
from copy import deepcopy

def argmin(values):
	return min(enumerate(values), key=lambda x: x[1])[0]
def avg(values):
	return float(sum(values))/len(values)

def readfile(filename):
	'''
	File format: Each line contains a comma separated list of real numbers, representing a single point.
	Returns a list of N points, where each point is a d-tuple.
	'''
	data = []
	with open(filename, 'r') as f:
		data = f.readlines()
	data = [tuple(map(float, line.split(','))) for line in data]
	return data

def writefile(filename, means):
	'''
	means: list of tuples
	Writes the means, one per line, into the file.
	'''
	if filename == None: return
	with open(filename, 'w') as f:
		for m in means:
			f.write(','.join(map(str, m)) + '\n')
	print ('Written means to file ' + filename)


def distance_euclidean(p1, p2):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point
	Returns the Euclidean distance b/w the two points.
	'''

	distance = sqrt(sum(map(lambda a: (a[0]-a[1])**2,zip(p1,p2))))

	# TODO [task1]:
	# Your function must work for all sized tuples.

	########################################
	return distance

def distance_manhattan(p1, p2):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point
	Returns the Manhattan distance b/w the two points.
	'''

	# k-means uses the Euclidean distance.
	# Changing the distant metric leads to variants which can be more/less robust to outliers,
	# and have different cluster densities. Doing this however, can sometimes lead to divergence!

	distance = sum(map(lambda a: abs(a[0]-a[1]),zip(p1,p2)))

	# TODO [task1]:
	# Your function must work for all sized tuples.

	########################################
	return distance

def initialization_forgy(data, k):
	'''
	data: list of tuples: the list of data points
	k: int: the number of cluster means to return
	Returns a list of tuples, representing the cluster means 
	'''

	means = []
	means = random.sample(data,k)

	# TODO [task1]:
	# Use the Forgy algorithm to initialize k cluster means.

	########################################
	assert len(means) == k
	return means

def initialization_randompartition(data, distance, k):
	'''
	data: list of tuples: the list of data points
	distance: callable: a function implementing the distance metric to use
	k: int: the number of cluster means to return
	Returns a list of tuples, representing the cluster means 
	'''

	means = []
	# Initialising the first mean randomly from the data
	rand_index = random.randint(0,len(data)-1)
	rand_point = data[rand_index]
	means.append(rand_point)
	
	num_done = 1	# The number of mean points so far
	# Iterating till we have k mean points
	while num_done < k:
		sum_tot = 0			# The sum of the distances so far
		point_distances = dict() # The square of the distance of a point to the nearest mean is stored in a dictionary
		for data_point in data :
			# Initialising the nearest mean and the nearest distance for each point
			nearest_mean = means[0]
			nearest_dist = distance(data_point,means[0])

# This is done separately because means[1] won't be valid if only one mean has been added so far
			if len(means) == 1: 
				point_distances[data_point] = nearest_dist**2 # Storing the square of the nearest distance
				sum_tot += nearest_dist**2	# Updating the sum so far
				continue
			# Going through all the means to determine the nearest mean for the data point
			for mean_point in means[1:]:
				dist = distance(data_point,mean_point)
				if dist < nearest_dist: #Updating the nearest mean and the nearest distance
					nearest_dist = dist 
					nearest_mean = mean_point

			point_distances[data_point] = nearest_dist**2 # Storing the square of the nearest distance
			sum_tot += nearest_dist**2 # Updating the sum so far

		for data_point in data:	# Normalizing the distribution
			point_distances[data_point] /= sum_tot*1.0

		# This is basically sampling from a multinomial distribution
		rand_num = random.random() # Generating the random number
		for data_point in data: # Checking if the random number lies in a 
# particular interval and if it does, we set the data point corresponding to that interval as our mean and we break off 
# we also check if the probability of that data point is not zero 
# (it can be zero only if it has been selected before, so this ensures that the same mean is not selected twice)
			if rand_num < point_distances[data_point] and point_distances[data_point] > 0:
				means.append(data_point)
				break
			rand_num -= point_distances[data_point] # This step is needed as this interval is done and we need to update the value for the next interval
		num_done += 1
	# TODO [task3]:
	# Use the kmeans++ algorithm to initialize k cluster means.
	# Make sure you use the distance function given as parameter.

	# NOTE: Provide extensive comments with your code.

	########################################
	assert len(means) == k
	return means

def initialization_kmeansplusplus(data, distance, k):
	'''
	data: list of tuples: the list of data points
	distance: callable: a function implementing the distance metric to use
	k: int: the number of cluster means to return
	Returns a list of tuples, representing the cluster means 
	'''

	means = []
	# Initialising the first mean randomly from the data
	rand_index = random.randint(0,len(data)-1)
	rand_point = data[rand_index]
	means.append(rand_point)
	
	num_done = 1	# The number of mean points so far
	# Iterating till we have k mean points
	while num_done < k:
		sum_tot = 0			# The sum of the distances so far
		point_distances = dict() # The square of the distance of a point to the nearest mean is stored in a dictionary
		for data_point in data :
			# Initialising the nearest mean and the nearest distance for each point
			nearest_mean = means[0]
			nearest_dist = distance(data_point,means[0])

# This is done separately because means[1] won't be valid if only one mean has been added so far
			if len(means) == 1: 
				point_distances[data_point] = nearest_dist**2 # Storing the square of the nearest distance
				sum_tot += nearest_dist**2	# Updating the sum so far
				continue
			# Going through all the means to determine the nearest mean for the data point
			for mean_point in means[1:]:
				dist = distance(data_point,mean_point)
				if dist < nearest_dist: #Updating the nearest mean and the nearest distance
					nearest_dist = dist 
					nearest_mean = mean_point

			point_distances[data_point] = nearest_dist**2 # Storing the square of the nearest distance
			sum_tot += nearest_dist**2 # Updating the sum so far

		for data_point in data:	# Normalizing the distribution
			point_distances[data_point] /= sum_tot*1.0

		# This is basically sampling from a multinomial distribution
		rand_num = random.random() # Generating the random number
		for data_point in data: # Checking if the random number lies in a 
# particular interval and if it does, we set the data point corresponding to that interval as our mean and we break off 
# we also check if the probability of that data point is not zero 
# (it can be zero only if it has been selected before, so this ensures that the same mean is not selected twice)
			if rand_num < point_distances[data_point] and point_distances[data_point] > 0:
				means.append(data_point)
				break
			rand_num -= point_distances[data_point] # This step is needed as this interval is done and we need to update the value for the next interval
		num_done += 1
	# TODO [task3]:
	# Use the kmeans++ algorithm to initialize k cluster means.
	# Make sure you use the distance function given as parameter.

	# NOTE: Provide extensive comments with your code.

	########################################
	assert len(means) == k
	return means


def iteration_one(data, means, distance):
	'''
	data: list of tuples: the list of data points
	means: list of tuples: the current cluster centers
	distance: callable: function implementing the distance metric to use
	Returns a list of tuples, representing the new cluster means after 1 iteration of k-means clustering algorithm.
	'''

	new_means = []
	k = len(means)
	dimension = len(data[0])

	def centroid(data_list):
		centroid = map(lambda a: sum(a)/(1.0*len(a)),zip(*data_list))
		return tuple(centroid)

	point_assignment = dict()
	for mean_point in means:
		point_assignment[mean_point] = []

	for data_point in data :
		nearest_mean = means[0]
		nearest_dist = distance(data_point,means[0])
		for mean_point in means[1:]:
			dist = distance(data_point,mean_point)
			if dist < nearest_dist:
				nearest_dist = dist
				nearest_mean = mean_point
		point_assignment[nearest_mean].append(data_point)

	for mean_point in means:
		if point_assignment[mean_point] == []:
			new_means.append(mean_point)
			continue
		new_mean = centroid(point_assignment[mean_point])
		new_means.append(new_mean)

	# TODO [task1]:
	# You must find the new cluster means.
	# Perform just 1 iteration (assignment+updation)

	########################################
	return new_means

def hasconverged(old_means, new_means, epsilon=1e-1):
	'''
	old_means: list of tuples: The cluster means found by the previous iteration
	new_means: list of tuples: The cluster means found by the current iteration
	Returns true iff no cluster center moved more than epsilon distance.
	'''

	converged = True
	for i in range(len(new_means)):
		if distance_euclidean(old_means[i],new_means[i]) > epsilon:
			converged = False
			break

	# TODO [task1]:
	# Use Euclidean distance to measure centroid displacements.

	########################################
	return converged



def iteration_many(data, means, distance, maxiter, epsilon=1e-1):
	'''
	maxiter: int: Number of iterations to perform
	Uses the iteration_one function.
	Performs maxiter iterations of the k-means clustering algorithm, and saves the cluster means of all iterations.
	Stops if convergence is reached earlier.
	Returns:
	all_means: list of (list of tuples): Each element of all_means is a list of the cluster means found by that iteration.
	'''

	all_means = []
	all_means.append(means)

	num_iter = 0
	not_converged = True
	curr_means = means[:]
	while num_iter < maxiter and not_converged:
		new_means = iteration_one(data,curr_means,distance)
		if hasconverged(curr_means,new_means,epsilon):
			not_converged = False
		all_means.append(new_means)
		curr_means = new_means[:]

	# TODO [task1]:
	# Make sure you've implemented the iteration_one, hasconverged functions.
	# Perform iterations by calling the iteration_one function multiple times.
	# Stop only if convergence is reached, or if max iterations have been exhausted.
	# Save the results of each iteration in all_means.
	# Tip: use deepcopy() if you run into weirdness.

	########################################

	return all_means



def performance_SSE(data, means, distance):

	'''
	data: list of tuples: the list of data points
	means: list of tuples: representing the cluster means 
	Returns: The Sum Squared Error of the clustering represented by means, on the data.
	'''
	sse = 0

	for data_point in data :
		nearest_mean = means[0]
		nearest_dist = distance(data_point,means[0])
		for mean_point in means[1:]:
			dist = distance(data_point,mean_point)
			if dist < nearest_dist:
				nearest_dist = dist
				nearest_mean = mean_point
		sse += distance(data_point,nearest_mean)**2

	# TODO [task1]:
	# Calculate the Sum Squared Error of the clustering represented by means, on the data.
	# Make sure to use the distance metric provided.

	########################################
	return sse



########################################################################
##                      DO NOT EDIT THE FOLLWOING                     ##
########################################################################


import sys
import argparse
import matplotlib.pyplot as plt
from itertools import cycle
from pprint import pprint as pprint

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', dest='input', type=str, help='Required. Dataset filename')
	parser.add_argument('-o', '--output', dest='output', type=str, help='Output filename')
	parser.add_argument('-iter', '--iter', '--maxiter', dest='maxiter', type=int, default=10000, help='Maximum number of iterations of the k-means algorithm to perform. (may stop earlier if convergence is achieved)')
	parser.add_argument('-e', '--eps', '--epsilon', dest='epsilon', type=float, default=1e-1, help='Minimum distance the cluster centroids move b/w two consecutive iterations for the algorithm to continue.')
	parser.add_argument('-init', '--init', '--initialization', dest='init', type=str, default='forgy', help='The initialization algorithm to be used. {forgy, randompartition, kmeans++}')
	parser.add_argument('-dist', '--dist', '--distance', dest='dist', type=str, default='euclidean', help='The distance metric to be used. {euclidean, manhattan}')
	parser.add_argument('-k', '--k', dest='k', type=int, default=5, help='The number of clusters to use.')
	parser.add_argument('-verbose', '--verbose', dest='verbose', type=bool, default=False, help='Turn on/off verbose.')
	parser.add_argument('-seed', '--seed', dest='seed', type=int, default=0, help='The RNG seed.')
	parser.add_argument('-numexperiments', '--numexperiments', dest='numexperiments', type=int, default=1, help='The number of experiments to run.')
	_a = parser.parse_args()

	if _a.input is None:
		print('Input filename required')
		parser.print_help()
		sys.exit(1)
	
	args = {}
	for a in vars(_a):
		args[a] = getattr(_a, a)

	if _a.init.lower() in ['random', 'randompartition']:
		args['init'] = initialization_randompartition
	elif _a.init.lower() in ['k++', 'kplusplus', 'kmeans++', 'kmeans', 'kmeansplusplus']:
		args['init'] = initialization_kmeansplusplus
	elif _a.init.lower() in ['forgy', 'frogy']:
		args['init'] = initialization_forgy
	else:
		print ('Unavailable initialization function.\n')
		parser.print_help()
		sys.exit(1)


	if _a.dist.lower() in ['manhattan', 'l1', 'median']:
		args['dist'] = distance_manhattan
	elif _a.dist.lower() in ['euclidean', 'euclid', 'l2']:
		args['dist'] = distance_euclidean
	else:
		print ('Unavailable distance metric.\n')
		parser.print_help()
		sys.exit(1)

	print ('-'*40 + '\n')
	print ('Arguments:')
	pprint(args)
	print ('-'*40 + '\n')
	return args

def visualize_data(data, all_means, args):
	print ('Visualizing...' )
	means = all_means[-1]
	k = args['k']
	distance = args['dist']
	clusters = [[] for _ in range(k)]
	for point in data:
		dlist = [distance(point, center) for center in means]
		clusters[argmin(dlist)].append(point)

	# plot each point of each cluster
	colors = cycle('rgbwkcmy')

	for c, points in zip(colors, clusters):
		x = [p[0] for p in points]
		y = [p[1] for p in points]
		plt.scatter(x,y, c = c)

	# plot each cluster centroid
	colors = cycle('krrkgkgr')
	colors = cycle('rgbkkcmy')

	for c, clusterindex in zip(colors, range(k)):
		x = [iteration[clusterindex][0] for iteration in all_means]
		y = [iteration[clusterindex][1] for iteration in all_means]
		plt.plot(x,y, '-x', c = c, linewidth='1', mew=15, ms=2)
	plt.axis('equal')
	plt.show()

def visualize_performance(data, all_means, distance):

	errors = [performance_SSE(data, means, distance) for means in all_means]
	plt.plot(range(len(all_means)), errors)
	plt.title('Performance plot')
	plt.xlabel('Iteration')
	plt.ylabel('Sum Squared Error')
	plt.show()


if __name__ == '__main__':

	args = parse()
	# Read data
	data = readfile(args['input'])
	print ('Number of points in input data: {}\n'.format(len(data)))
	verbose = args['verbose']

	totalSSE = 0
	totaliter = 0

	for experiment in range(args['numexperiments']):
		print ('Experiment: {}'.format(experiment+1))
		random.seed(args['seed'] + experiment)
		print ('Seed: {}'.format(args['seed'] + experiment))

		# Initialize means
		means = []
		if args['init'] == initialization_forgy:
			means = args['init'](data, args['k']) # Forgy doesn't need distance metric
		else:
			means = args['init'](data, args['dist'], args['k'])

		if verbose:
			print ('Means initialized to:')
			print (means)
			print ('')

		# Run k-means clustering
		all_means = iteration_many(data, means, args['dist'], args['maxiter'], args['epsilon'])

		SSE = performance_SSE(data, all_means[-1], args['dist'])
		totalSSE += SSE
		totaliter += len(all_means)-1
		print ('Sum Squared Error: {}'.format(SSE))
		print ('Number of iterations till termination: {}'.format(len(all_means)-1))
		print ('Convergence achieved: {}'.format(hasconverged(all_means[-1], all_means[-2])))


		if verbose:
			print ('\nFinal means:')
			print (all_means[-1] )
			print ('')
			
	print ('\n\nAverage SSE: {}'.format(float(totalSSE)/args['numexperiments']))
	print ('Average number of iterations: {}'.format(float(totaliter)/args['numexperiments']))

	if args['numexperiments'] == 1:
		# save the result
		writefile(args['output'], all_means[-1])

		# If the data is 2-d and small, visualize it.
		if len(data) < 5000 and len(data[0]) == 2:
			visualize_data(data, all_means, args)

		visualize_performance(data, all_means, args['dist'])