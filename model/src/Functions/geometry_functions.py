'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Geometry functions  --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
Set of fonctions for diverse computations on the environment's geometry
'''

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def convert_polar_to_cartesian (polar_vector) : 
	
	# Converts a polar vector (rho, theta) in a cartesian vector

	cartesian_vector = [polar_vector[0] * np.cos(polar_vector[1]) , polar_vector[0] * np.sin(polar_vector[1])]
	return(cartesian_vector)

def normalize_matrix_by_row (matrix) : 
	
	# Normalizes a matrix such that the sum of the value on each row equals 1

	nrow,ncol = np.shape(matrix)
	normalized_matrix = np.empty((nrow,ncol))
	for row in range (nrow) : 
		row_sum  = np.sum(matrix[row,:])
		if row_sum !=0 : 
			normalized_matrix[row,:]=matrix[row,:]/row_sum
	return(normalized_matrix)

def distance(point1,point2) : 
	
	# Gives the euclidean distance between two points in a 2D space.

	return(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

def get_matrix_of_distances_between_flowers(array_geometry) : 
	
	# Given the position of flowers, computes the matrix of pairwise distances between flowers

	matrix_of_coordinates = array_geometry[["x","y"]] # keep the coordinates
	matrix_of_pairwise_distances = euclidean_distances (matrix_of_coordinates,matrix_of_coordinates)
	return(matrix_of_pairwise_distances)

def give_probability_of_vector_with_dist_factor(x,dist_factor) :
	
	# Gets the probability of x given the function [probability = 1/x^distFactor]

	dimensions = np.shape(x)
	probabilities = np.zeros(dimensions)
	for i in range (dimensions[0]) : 
		for j in range (dimensions[1]) : 
			if x[i,j] != 0 : probabilities[i,j] = 1/x[i,j]**dist_factor
	return(probabilities)


def get_coordinates(timesteps_left_to_wait,timestep_to_next_flower,coordinates_previous_flower,coordinates_next_flower) : 

	# Get the current coordinates of the bee given the current time, the starting and the end position

	p = timesteps_left_to_wait/timestep_to_next_flower
	coordinates = p*np.array(coordinates_previous_flower) + (1-p)*np.array(coordinates_next_flower)

	return(coordinates)

def get_stoppoint_coordinates(distance_travelled,distance_with_penalty,current_flower,chosen_flower,max_distance_travelled,array_geometry,
	velocity_of_bee,number_of_seconds_per_timesteps) : 

	# Finds the coordinates of the stoppoint (ie the point at which the bee goes back to the nest)

	real_distance_to_be_travelled = max_distance_travelled - distance_travelled
	proportion_to_be_travelled = real_distance_to_be_travelled/distance_with_penalty

	current_flower_coordinates = np.array((array_geometry.loc[current_flower,"x"],array_geometry.loc[current_flower,"y"]))
	chosen_flower_coordinates = np.array((array_geometry.loc[chosen_flower,"x"],array_geometry.loc[chosen_flower,"y"]))

	stoppoint_coordinates = (1-proportion_to_be_travelled)*current_flower_coordinates + proportion_to_be_travelled*chosen_flower_coordinates

	timesteps_to_stoppoint = np.ceil(real_distance_to_be_travelled/(velocity_of_bee*number_of_seconds_per_timesteps))

	return(stoppoint_coordinates,timesteps_to_stoppoint)

def get_flowers_perceived(number_of_flowers,coordinates_start_flower,coordinates_end_flower,array_geometry,perception_range) : 

	# Get the list of flowers percieved on the path of the bee
	x_start, y_start = coordinates_start_flower
	x_end, y_end = coordinates_end_flower

	list_of_flower_discovered = []

	if x_start != x_end or y_start != y_end: 

		# parameter of the line from start_flower to end_flower (ax+by+c=0)
		a = y_end - y_start
		b = x_start - x_end
		c = y_start*x_end - y_end*x_start

		# coordinates of the vector from start_flower to end_flower
		start_to_end_vector = np.array([x_end-x_start,y_end-y_start])

		for flower in range (number_of_flowers) : 

			x,y = array_geometry.loc[flower,"x"], array_geometry.loc[flower,"y"]

			start_to_flower_vector = np.array([x-x_start, y-y_start])
			end_to_flower_vector = np.array([x-x_end,y-y_end])

			# test if flower is before the start of the segment :
			if np.dot(start_to_end_vector,start_to_flower_vector) < 0 : 
				distance_to_trajectory = distance((x,y),(x_start,y_start))

			# test if flower is after the end of the segment :
			elif np.dot(-start_to_end_vector,end_to_flower_vector) < 0 : 
				distance_to_trajectory = distance((x,y),(x_end,y_end))

			else : 
				distance_to_trajectory = np.abs(a*x + b*y + c)/np.sqrt(a*a + b*b)

			if distance_to_trajectory <= perception_range :
				 list_of_flower_discovered.append(flower)
	
	return(list_of_flower_discovered)

def get_matrix_of_flowers_perceived(array_geometry, perception_range) :

	# Defines the flowers perceived during the flight between two flowers for all pairs of flowers

	number_of_flowers = len(array_geometry.index)
	matrix_of_flowers_perceived = np.full((number_of_flowers,number_of_flowers),None)

	for start_flower in range (number_of_flowers) : 
		for end_flower in range (number_of_flowers) : 
			x_start, y_start = array_geometry.loc[start_flower,"x"], array_geometry.loc[start_flower,"y"]
			x_end, y_end = array_geometry.loc[end_flower,"x"], array_geometry.loc[end_flower,"y"]
			matrix_of_flowers_perceived[start_flower,end_flower] = get_flowers_perceived(number_of_flowers,(x_start, y_start),(x_end, y_end),array_geometry,perception_range)


	return(matrix_of_flowers_perceived)