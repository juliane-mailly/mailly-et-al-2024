'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''
# Simulation functions --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
This piece of code contains all the functions necessary to run all simulations with a given set of parameters and generate raw outputs.
This code is object-oriented; Three classes are used to complete a simulation (Bee, Flower, Environment). 
'''

import numpy as np
import pandas as pd
import copy
import other_functions
import geometry_functions
import data_analysis_functions

class Bee:

	# Defines all necessary functions and attributes for bees.

	def __init__ (self, ID, individual_bee_info, initial_Q_table, number_of_flowers):

		# Initialize the current state of the bee
		self.is_recruited = False
		self.is_foraging = False
		self.foraging_mode = None
		self.is_discovering = False
		self.is_homing = False
		self.is_in_nest = True
		self.output_position = 0
		self.timpesteps_next_flower = 0
		self.distance_next_flower = 0.
		self.goes_to_stopppoint = False
		self.comes_from_stoppoint = False
		self.stoppoint_coordinates = None
		self.coordinates = np.array((0.,0.))

		# Initialize the data of the current bout
		self.bout_index = -1
		self.bout_duration = 0.
		self.number_of_resources_foraged = 0
		self.distance_travelled = 0.
		self.visit_sequence = [0]
		self.interactions_occurrences = 0
		self.path_length = 0. 
		self.best_flower_encountered = 0
		self.best_nectar_reward_encountered = 0.

		self.ID = ID
		for (parameter_name,parameter_value) in individual_bee_info.items():
			setattr(self,parameter_name,parameter_value)		
		self.waiting_timesteps = np.random.randint(low=0,high=self.timesteps_inactive_in_nest+1)
		self.working_memory = np.zeros(number_of_flowers)
		if self.allowing_discovery : self.memory = np.array([1] + list(np.zeros(number_of_flowers-1)))
		else : self.memory = np.ones(number_of_flowers)
		self.Q_table = copy.deepcopy(initial_Q_table)



	def reboot_bout_data(self):

		# Used to reboot bee data at the end of a bout

		self.bout_duration = 0.
		self.number_of_resources_foraged = 0
		self.distance_travelled = 0.
		self.visit_sequence = [0]
		self.interactions_occurrences = 0
		self.path_length = 0. 
		self.best_flower_encountered = 0
		self.best_nectar_reward_encountered = 0.



	def learning(self, previous_flower, current_flower, nectar_reward, probabilities_discovering_flowers):

		# Learning function that updates the bee's Q-table

		if nectar_reward is not None : 

			reward  = nectar_reward*probabilities_discovering_flowers[previous_flower,current_flower]

			delta = reward+self.gamma*np.max(self.Q_table[current_flower,:])-self.Q_table[previous_flower,current_flower]

			if self.is_recruited :			
				self.Q_table[previous_flower,current_flower] = self.Q_table[previous_flower,current_flower] + self.social_learning_rate*delta

			else : 				
				self.Q_table[previous_flower,current_flower] = self.Q_table[previous_flower,current_flower] + self.alpha*delta



	def initialize_Q_values_of_new_flower(self, capacity_of_flower, probabilities_discovering_flowers, new_flower) :
  		
		# Initialise a new flower's values in a bee's Q-table

		self.Q_table[:, new_flower] = (capacity_of_flower/2)*probabilities_discovering_flowers[:,new_flower]



	def decay_memory(self,number_of_seconds_per_timesteps) :

		# Decay of the memory through time

		# Working memory
		self.working_memory = self.working_memory - 1
		self.working_memory = np.maximum(self.working_memory,0)

		# Long-term memory (not used in this article)
		if self.allowing_discovery and self.allowing_memory_decay :
			self.memory[1:] = self.memory[1:] - 1/(self.memory_span/number_of_seconds_per_timesteps)
			self.memory = np.maximum(self.memory,0)



	def choose_if_discovering(self,capacity_of_flower):

		# Defines if the bee will be exploiting its knowledge or exploring the environment (= discovering a new flower)
		# (not used in this article)

		number_of_flowers_not_in_memory = np.sum(self.memory==0)

		if number_of_flowers_not_in_memory == 0 :
			self.is_discovering = False

		else : 
			sum_memory = np.sum(self.memory)
			
			# Exponential decay of the probability of discovering flowers as the memory gets more full
			decay_factor = - np.log(self.proba_discovery_when_max_crop)/(self.max_crop_capacity/capacity_of_flower)
			probability_discovery = np.exp(-decay_factor*(sum_memory-1))

			self.is_discovering = np.random.binomial(n=1,p=probability_discovery)



	def choose_recruit(self, bees_in_nest) :
		
	# Defines if the bee will be recruiting a bee in the nest and the ID of said bee
	# (not used in this article)

		recruitment = (np.random.binomial(n=1,p=self.probability_of_recruitment)) and np.sum(bees_in_nest)!=0 and (self.best_flower_encountered!=0)
		if recruitment : 
			recruit_ID = np.random.choice(np.where(bees_in_nest)[0])
		else : 
			recruit_ID = None
		return(recruit_ID)



	def recruitment(self, best_flower_encountered, best_nectar_reward_encountered,capacity_of_flower,
				 probabilities_discovering_flowers, number_of_flowers):
		

		# Deals with sharing of social information during recruitment
		# (not used in this article)

		if self.memory[best_flower_encountered] == 0 : 
			self.initialize_Q_values_of_new_flower(capacity_of_flower, probabilities_discovering_flowers, 
										  best_flower_encountered)

		for flower in range (number_of_flowers) : 
			self.learning(flower, best_flower_encountered, best_nectar_reward_encountered, probabilities_discovering_flowers)

		self.memory[best_flower_encountered] = 1
		self.waiting_timesteps = 0			



	def get_available_destinations(self):

		# Gives the destinations that are available for the bee

		if self.is_discovering : 
			available_destinations = np.where(self.memory==0)[0]
		
		else :

			available_destinations = []
			for flower_ID in range (len(self.memory)) : 
				if np.random.binomial(n=1,p=self.memory[flower_ID]) : 
						available_destinations.append(flower_ID)

		omitted_destinations = np.unique(np.where(self.working_memory>0)[0])
		
		if self.force_full_crop and 0 not in omitted_destinations :
			omitted_destinations = np.concatenate(([0], omitted_destinations))

		for omitted_destination in omitted_destinations : 
			available_destinations = np.delete(available_destinations, np.where(available_destinations==omitted_destination))


		return(available_destinations)



	def choose_next_position(self, probabilities_discovering_flowers, number_of_flowers):

		# Makes the decision on which flower to go next

		# Get available destination for this bee
		available_destinations = self.get_available_destinations()
		destination_is_available = [(flower in available_destinations) for flower in range (number_of_flowers)]


		# Get the probabilities
		if not self.is_discovering : 
			Q_values = self.Q_table[self.visit_sequence[-1],destination_is_available] #HERE
			probabilities = other_functions.softmax(Q_values,self.beta)
		else :
			probabilities = probabilities_discovering_flowers[self.visit_sequence[-1],destination_is_available]
			sum_of_probabilities = np.sum(probabilities)
			if sum_of_probabilities !=0 : probabilities = probabilities/sum_of_probabilities


		# Choose the next destination
		if len(probabilities)==0 or np.sum(probabilities)==0: # The bee does not have any possibility
			chosen_flower = 0 # Go to the nest

		else : # There is at least one potential destination
			chosen_flower = np.random.choice(a=available_destinations,p=probabilities)
			
		return(chosen_flower)



	def discovering_flowers(self, capacity_of_flower, probabilities_discovering_flowers,matrix_of_flowers_perceived,
						 array_geometry, number_of_flowers):

		# Deals with discovering new flowers, notably the flowers that are perceived nearby the bees
		# (not used in this article)

		if self.allowing_discovery : 

			if self.comes_from_stoppoint : # comes from the stoppoint and goes to the nest
				
				coordinates_next_flower = np.array((0,0))
				flowers_perceived = geometry_functions.get_flowers_perceived(number_of_flowers,self.stoppoint_coordinates,coordinates_next_flower,
					array_geometry, self.perception_range)

			elif self.goes_to_stopppoint : # goes to the stoppoint
				coordinates_previous_flower = np.array((array_geometry.loc[self.visit_sequence[-2],"x"],array_geometry.loc[self.visit_sequence[-2],"y"])) 
				flowers_perceived = geometry_functions.get_flowers_perceived(number_of_flowers,coordinates_previous_flower,self.stoppoint_coordinates,
					array_geometry, self.perception_range)

			else : 
				flowers_perceived = matrix_of_flowers_perceived[self.visit_sequence[-2],self.visit_sequence[-1]]

			for flower in flowers_perceived : 

				if self.memory[flower] == 0 : # The flower is not known
					self.memory[flower] = 1
					# Complete the learning array of the bee
					self.initialize_Q_values_of_new_flower(capacity_of_flower, probabilities_discovering_flowers, flower)


	def update_coordinates_and_output_position(self, previous_flower, current_flower):

		# Deals with the position of the bee at each timestep. Computes the exact position between flowers
		# It also accounts for when the bee goes home from tiredness in the middle of a flight (the breaking point is called stopppoint)

		if self.is_in_nest : 
			self.coordinates = np.array((0,0))
			self.output_position = 0

		if self.is_homing : 

			if self.waiting_timesteps != 0 and self.goes_to_stopppoint : # still needs to fly toward the stopppoint
				self.output_position = None
				self.coordinates = geometry_functions.get_coordinates(self.waiting_timesteps,self.timpesteps_next_flower,
					previous_flower.coordinates,self.stoppoint_coordinates) 


			elif self.waiting_timesteps == 0 and self.goes_to_stopppoint : # Has found the stoppoint		
				self.output_position = None	
				self.coordinates = self.stoppoint_coordinates


			elif self.waiting_timesteps != 0 and self.comes_from_stoppoint : # is going back to the nest from the stopppoint
				self.output_position = None
				self.coordinates = geometry_functions.get_coordinates(self.waiting_timesteps,self.timpesteps_next_flower,
						self.stoppoint_coordinates,current_flower.coordinates) 

			elif self.waiting_timesteps != 0 : # going back to nest but not from the stoppoint
				self.output_position = None
				self.coordinates = geometry_functions.get_coordinates(self.waiting_timesteps,self.timpesteps_next_flower,
						previous_flower.coordinates,current_flower.coordinates)

			else: # has arrived to the nest
				self.output_position = 0
				self.coordinates = np.array((0,0))

		if self.is_foraging :

			if self.foraging_mode == "flying" : 

				if self.waiting_timesteps > 0 : # still needs to fly:
					self.output_position = None
					self.coordinates = geometry_functions.get_coordinates(self.waiting_timesteps,self.timpesteps_next_flower,
						previous_flower.coordinates,current_flower.coordinates) 

				else : # has arrived to destination and will feed
				
					# Arriving 
					self.output_position = current_flower.ID
					self.coordinates = current_flower.coordinates


			if self.foraging_mode == "feeding" or self.foraging_mode == "choosing" :

				self.output_position = current_flower.ID
				self.coordinates = current_flower.coordinates



class Flower:

	# Defines the necessary attributes and functions for the flowers in the environment

	def __init__(self, ID, time_constants, coordinates, patch, time_flower_replenishment):

		# Initialise the flower's information
		self.ID = ID
		self.capacity = time_constants["capacity_of_flower"]
		if time_flower_replenishment == -1 : self.number_of_timesteps_replenishment = np.inf 
		else : self.number_of_timesteps_replenishment =  int(time_flower_replenishment/time_constants["number_of_seconds_per_timesteps"])
		self.resource_replenished_per_timestep = time_constants["capacity_of_flower"]/self.number_of_timesteps_replenishment
		if self.ID == 0 :
			self.nectar = 0
			self.is_occupied = True
		else : 
			self.nectar = self.capacity
			self.is_occupied = False
		self.coordinates = coordinates
		self.patch = patch

	def replenish(self):

		# Replenishes the flower's nectar through time until reaching its maximal capacity

		if not self.is_occupied : 
			if self.nectar + self.resource_replenished_per_timestep <= self.capacity : 
				self.nectar = self.nectar + self.resource_replenished_per_timestep
			else :
				self.nectar = self.capacity


class Environment:

	# Defines the necessary attributes and functions of the environment to run a simulation

	def __init__(self, number_of_bees, time_constants, environment_matrices, bee_info, initial_Q_table, array_geometry):

		# Initialise the environment's information

		self.timestep = 0
		self.time = 0.

		for (parameter_name,parameter_value) in time_constants.items():
			setattr(self,parameter_name,parameter_value)
		for (parameter_name,parameter_value) in environment_matrices.items():
			setattr(self,parameter_name,parameter_value)

		self.number_of_bees = number_of_bees
		self.number_of_flowers = len(array_geometry.index)

		self.array_geometry = array_geometry

		# List of outputs

		self.list_of_visit_sequences=[]
		self.list_of_interaction_occurences = []
		self.list_of_bout_durations = []
		self.quantity_of_nectar_per_bout = []
		self.positions_at_each_timestep = np.full((self.number_of_timesteps_per_sim,number_of_bees),None)
		self.list_arrival_times = []
		self.coordinates_at_each_timestep = np.full((self.number_of_timesteps_per_sim,number_of_bees,2),0.)
		self.memory_at_each_timestep = np.full((self.number_of_timesteps_per_sim*number_of_bees,self.number_of_flowers),0.)
		self.working_memory_at_each_timestep = np.full((self.number_of_timesteps_per_sim*number_of_bees,self.number_of_flowers),0.)

		# Initialise the bees
		self.bees = [Bee(i, bee_info.iloc[i,:], initial_Q_table, self.number_of_flowers) for i in range (number_of_bees)]
		self.bees_in_nest = [True for i in range (number_of_bees)]

		# Initialise the flowers
		self.list_time_flower_replenishment = [-1] + [time_constants["time_flower_replenishment"]]*(self.number_of_flowers-1)
		self.flowers = [Flower(i,time_constants,np.array((array_geometry.iloc[i,1:3])),array_geometry.iloc[i,3], self.list_time_flower_replenishment[i]) for i in range (self.number_of_flowers)]

	def time_passes(self) : 

		# Allows the necessary steps of time passing (replenishment of flowers, decay of bees' memory)
		self.timestep = self.timestep + 1
		self.time = self.time + self.number_of_seconds_per_timesteps

		for flower in self.flowers : 
			flower.replenish()

		for bee in self.bees : 
			bee.decay_memory(self.number_of_seconds_per_timesteps)
			if bee.waiting_timesteps > 0 : bee.waiting_timesteps = bee.waiting_timesteps - 1
			if not bee.is_in_nest : bee.bout_duration = bee.bout_duration + self.number_of_seconds_per_timesteps

	def update_outputs(self, bee): 

		# Update the list of outputs throughout the simulation

		# List outputs
		if len(self.list_of_visit_sequences) == bee.bout_index : # Need to add another line to the list

			# Visitation sequence
			sequence_for_each_bee = [None for i in range (self.number_of_bees)]
			sequence_for_each_bee[bee.ID] = bee.visit_sequence
			self.list_of_visit_sequences = self.list_of_visit_sequences + [sequence_for_each_bee]
			# Competition
			competition_for_each_bee = [None for i in range (self.number_of_bees)]
			competition_for_each_bee[bee.ID] = bee.interactions_occurrences
			self.list_of_interaction_occurences = self.list_of_interaction_occurences + [competition_for_each_bee]
			# Bout duration
			bout_duration_each_bee = [None for i in range (self.number_of_bees)]
			bout_duration_each_bee[bee.ID] = bee.bout_duration
			self.list_of_bout_durations = self.list_of_bout_durations + [bout_duration_each_bee]
			# Resources foraged
			resources_each_bee = [None for i in range (self.number_of_bees)]
			resources_each_bee[bee.ID] = bee.number_of_resources_foraged
			self.quantity_of_nectar_per_bout = self.quantity_of_nectar_per_bout + [resources_each_bee]

		else : 


			# Visitation sequence
			self.list_of_visit_sequences[bee.bout_index][bee.ID] = bee.visit_sequence
			# Competition
			self.list_of_interaction_occurences[bee.bout_index][bee.ID] = bee.interactions_occurrences
			# Bout duration
			self.list_of_bout_durations[bee.bout_index][bee.ID] = bee.bout_duration
			# Resources foraged
			self.quantity_of_nectar_per_bout[bee.bout_index][bee.ID] = bee.number_of_resources_foraged

	def update_bee(self, bee): 

		# Update the bees' status at each timestep


		if len(bee.visit_sequence) > 1 : 
			previous_flower = self.flowers[bee.visit_sequence[-2]]
		else : 
			previous_flower = None	
		current_flower = self.flowers[bee.visit_sequence[-1]]
		

		bee.update_coordinates_and_output_position(previous_flower,current_flower)

		if bee.is_in_nest :

			if bee.waiting_timesteps == 0 : 
				bee.is_in_nest = False
				self.bees_in_nest[bee.ID] = False
				bee.is_foraging = True
				bee.foraging_mode = "choosing"
				bee.bout_index = bee.bout_index + 1

		elif bee.is_homing : 

			if bee.waiting_timesteps == 0 :
				
				if bee.goes_to_stopppoint : # Has found the stoppoint = breakpoint at which the bee goes home

					bee.distance_travelled = bee.max_distance_travelled # The bee has reached the maximum distance

					bee.discovering_flowers(self.capacity_of_flower, self.probabilities_discovering_flowers,
							 self.matrix_of_flowers_perceived,self.array_geometry, self.number_of_flowers)

					bee.goes_to_stopppoint = False # Just a code to mark that the bee went to stoppoint
					bee.comes_from_stoppoint = True # Goes back to nest now

					bee.distance_next_flower = geometry_functions.distance(bee.stoppoint_coordinates,[0,0])
					bee.timpesteps_next_flower = np.ceil(bee.distance_next_flower/(bee.velocity*self.number_of_seconds_per_timesteps))
					bee.waiting_timesteps = max(bee.timpesteps_next_flower-1,0) 

				else : # The bee has arrived to the nest

					# Update distance
					bee.distance_travelled = bee.distance_travelled + bee.distance_next_flower
					bee.path_length = bee.path_length + self.distance_between_flowers[bee.visit_sequence[-2],bee.visit_sequence[-1]]

					# Discover flowers
					bee.discovering_flowers(self.capacity_of_flower, self.probabilities_discovering_flowers, 
							 self.matrix_of_flowers_perceived, self.array_geometry, self.number_of_flowers)

					bee.comes_from_stoppoint = False

					# Learn transition (remark: if stoppoint, bee will learn transition between previous flower and nest, not between previous flower and intended next flower)
					nectar_reward = 0.
					bee.learning(bee.visit_sequence[-2], bee.visit_sequence[-1], nectar_reward, self.probabilities_discovering_flowers)

					# Recruitement
					recruit_ID = bee.choose_recruit(self.bees_in_nest)
					if recruit_ID is not None : # Recruitment is happening
						recruit = self.bees[recruit_ID]
						recruit.is_recruited = True
						recruit.recruitment(bee.best_flower_encountered, bee.best_nectar_reward_encountered,
						  self.capacity_of_flower,self.probabilities_discovering_flowers, self.number_of_flowers)
						recruit.is_recruited = False

					# Update the outputs
					self.update_outputs(bee)

					# Reinitialize the values
					bee.reboot_bout_data()

					# Change status of bee
					bee.is_homing = False
					bee.is_in_nest = True
					self.bees_in_nest[bee.ID] = True

					bee.waiting_timesteps = bee.timesteps_inactive_in_nest

		elif bee.is_foraging :

			if bee.foraging_mode == "flying" :

				if bee.waiting_timesteps == 0  : # The bee has arrived to destination and will feed

					# Arriving 
					self.list_arrival_times.append([self.timestep, self.time, bee.ID, current_flower.ID])

					bee.distance_travelled = bee.distance_travelled + bee.distance_next_flower
					bee.path_length = bee.path_length + self.distance_between_flowers[previous_flower.ID,current_flower.ID]

					# Update memory and Q table of discovered flowers
					bee.discovering_flowers(self.capacity_of_flower, self.probabilities_discovering_flowers, 
							 self.matrix_of_flowers_perceived, self.array_geometry, self.number_of_flowers)

					# If flower is free, will go in "feeding" mode		
					if not current_flower.is_occupied : 
						# Change bee mode to "feeding"
						bee.foraging_mode = "feeding"
						# Compute the time that the bee will take to feed on the flower
						bee.waiting_timesteps = max(np.ceil(current_flower.nectar/bee.quantity_nectar_eaten_per_timestep) -1, 0)
						# Notify that the flower is now occupied by a bee (if it is not the nest)
						if current_flower.ID != 0 : current_flower.is_occupied = True

					# If flower is already taken by another bee
					else : 
						# Learn the transition (no reward)
						nectar_reward = 0.
						bee.learning(previous_flower.ID, current_flower.ID, nectar_reward, self.probabilities_discovering_flowers)
						# Update competition
						bee.interactions_occurrences = bee.interactions_occurrences + 1
						# Change mode to "choosing"
						bee.foraging_mode = "choosing"


			elif bee.foraging_mode == "feeding" :

				if bee.waiting_timesteps == 0 : # The bee is done eating

					# Eating
					nectar_reward = current_flower.nectar
					nectar_eaten = min(nectar_reward, bee.max_crop_capacity-bee.number_of_resources_foraged)
					bee.number_of_resources_foraged = bee.number_of_resources_foraged + nectar_eaten  
					current_flower.nectar = current_flower.nectar - nectar_eaten  # depleting flower


					# Learning
					bee.learning(previous_flower.ID, current_flower.ID, nectar_reward, self.probabilities_discovering_flowers)

					# Update best flower encountered by bee
					if nectar_reward > bee.best_nectar_reward_encountered : 
						bee.best_nectar_reward_encountered = nectar_reward
						bee.best_flower_encountered = current_flower.ID

					# Change bee mode to "choosing"
					bee.foraging_mode = "choosing"

					# Notify that flower is now free
					current_flower.is_occupied = False

			elif bee.foraging_mode == "choosing" :

				# Put current flower in working memory
				bee.working_memory[current_flower.ID] =  np.ceil(bee.working_memory_span/self.number_of_seconds_per_timesteps)

				# Check if bout is finished: 
				if bee.number_of_resources_foraged >= bee.max_crop_capacity :
					bee.is_discovering = False
					bee.is_foraging = False
					bee.is_homing = True 
					chosen_flower = 0

				else : # Choose next position
					bee.choose_if_discovering(self.capacity_of_flower)
					chosen_flower = bee.choose_next_position(self.probabilities_discovering_flowers, self.number_of_flowers)

					# If allow nest return and the bee has chosen the nest, or if the bout has ended for this bee, go back to the nest 
					if chosen_flower == 0:
						bee.is_foraging = False
						bee.is_homing = True

					else : 
						# Change bee mode to "flying"
						bee.foraging_mode = "flying"


				# Compute distance to be travelled and timesteps left to wait
				if not bee.is_discovering: distance_penalty = 1.
				else : distance_penalty = np.random.uniform(low=1.,high=bee.cost_of_discovery)
				bee.distance_next_flower = self.distance_between_flowers[current_flower.ID,chosen_flower]*distance_penalty

				# If distance is too much and the bee is not going back to nest, force to go back to nest and compute the stop point
				if (not bee.force_full_crop) and (bee.max_distance_travelled < bee.distance_travelled + bee.distance_next_flower) and (chosen_flower != 0): # the bee is going to outdo the max distance and it was not going back to the nest

					bee.is_foraging = False
					bee.is_homing = True
					bee.goes_to_stopppoint = True
					bee.stoppoint_coordinates,bee.timpesteps_next_flower = geometry_functions.get_stoppoint_coordinates(bee.distance_travelled,
						bee.distance_next_flower,current_flower.ID,chosen_flower,bee.max_distance_travelled,self.array_geometry,
						bee.velocity,self.number_of_seconds_per_timesteps) 
					chosen_flower = -1 # code to signal that will go to a stoppoint before coming back to nest

				else :
					if distance_penalty == 1 :
						bee.timpesteps_next_flower = self.timesteps_between_flowers[current_flower.ID,chosen_flower]
					else : 
						bee.timpesteps_next_flower = np.ceil(bee.distance_next_flower/(bee.velocity*self.number_of_seconds_per_timesteps))
					
				bee.waiting_timesteps = max(bee.timpesteps_next_flower-1,0)


				# Update position and previous/current/next_flower
				if chosen_flower == -1 : 
					bee.visit_sequence = bee.visit_sequence + [0]
				else : 
					bee.visit_sequence = bee.visit_sequence + [chosen_flower]

		self.positions_at_each_timestep[self.timestep,bee.ID] = bee.output_position
		self.coordinates_at_each_timestep[self.timestep,bee.ID,:] = bee.coordinates
		self.memory_at_each_timestep[self.timestep*self.number_of_bees+bee.ID,:] = bee.memory
		self.working_memory_at_each_timestep[self.timestep*self.number_of_bees+bee.ID,:] = bee.working_memory

	def run_one_simulation(self):


		while self.timestep < self.number_of_timesteps_per_sim :

			bee_order = [i for i in range (self.number_of_bees)]
			np.random.shuffle(bee_order)

			for bee_ID in bee_order : 
				bee = self.bees[bee_ID]
				self.update_bee(bee)

			self.time_passes()

		self.max_number_of_bouts = np.max([bee.bout_index for bee in self.bees])





def run_all_simulations_and_make_output_files(initial_Q_table,environment_matrices, number_of_simulations,time_constants,number_of_bees,bee_info,
	array_geometry,output_folder_of_sim,create_video_output,number_of_simulations_to_show):

	# Final loop on all the simulations. Generates outputs.

	# Retrieve time constants
	number_of_seconds_per_timesteps = time_constants["number_of_seconds_per_timesteps"]
	number_of_timesteps_per_sim = time_constants["number_of_timesteps_per_sim"]

	# Initialize output structures

	matrix_of_visit_sequences = [] # sim, bout, bee, visit_sequence
	matrix_of_similarity_1 = [] # sim, bout, bee, route_similarity
	matrix_of_similarity_2 = [] # sim, bout, bee, route_similarity
	matrix_of_similarity_3 = [] # sim, bout, bee, route_similarity
	matrix_of_interactions = [] # sim, bout, bee, competition
	matrix_of_bout_durations = [] # sim, bout, bee, bout_duration
	matrix_of_positions = [] # sim, timestep, time, bee, position
	matrix_of_arrival_times = [] # sim, timestep, time, bee, flower
	matrix_of_resources_foraged = [] # sim, bout, bee, quantity_of_nectar
	#matrix_of_coordinates = [] # sim, timestep, time, bee, x, y
	maximum_length_of_a_route = 0

	# Sim loop

	for sim in range (number_of_simulations): 

		# Initialize simulation objects
		environment = Environment(number_of_bees, time_constants, environment_matrices, bee_info, initial_Q_table, array_geometry)
		environment.run_one_simulation()

		# Video output
		create_video_for_this_sim = create_video_output and sim < number_of_simulations_to_show

		if create_video_for_this_sim : 
			gif_name = "sim"+str(sim)+'.gif'
			other_functions.create_video_output(gif_name,array_geometry,environment.coordinates_at_each_timestep,
				number_of_bees,time_constants,output_folder_of_sim,True,
				environment.memory_at_each_timestep)
		

		#other_functions.visualize_trajectory(environment.bees[0].working_memory_span,array_geometry,environment.coordinates_at_each_timestep,number_of_bees,output_folder_of_sim)


		# Similarity index = 1
		list_of_route_similarities_1 = data_analysis_functions.compute_route_similarity_of_sim(environment.list_of_visit_sequences, 1)

		# Similarity index = 2
		list_of_route_similarities_2 = data_analysis_functions.compute_route_similarity_of_sim(environment.list_of_visit_sequences, 2)

		# Similarity index = 3
		list_of_route_similarities_3 = data_analysis_functions.compute_route_similarity_of_sim(environment.list_of_visit_sequences, 3)

		for bout in range (environment.max_number_of_bouts) : 

			for bee in range (number_of_bees) : 

				# Visitation sequences
				if environment.list_of_visit_sequences[bout][bee] is None : environment.list_of_visit_sequences[bout][bee] = []
				matrix_of_visit_sequences.append([sim,bout,bee]+environment.list_of_visit_sequences[bout][bee])

				# Similarity index = 1
				if bout < (environment.max_number_of_bouts - 1) : 
					matrix_of_similarity_1.append([sim,bout,bee,list_of_route_similarities_1[bout][bee]])

				# Similarity index = 2
				if bout < (environment.max_number_of_bouts - 2) : 
					matrix_of_similarity_2.append([sim,bout,bee,list_of_route_similarities_2[bout][bee]])

				# Similarity index = 3
				if bout < (environment.max_number_of_bouts - 3) : 
					matrix_of_similarity_3.append([sim,bout,bee,list_of_route_similarities_3[bout][bee]])

				# Interactions
				matrix_of_interactions.append([sim,bout,bee,environment.list_of_interaction_occurences[bout][bee]])

				# Bout duration
				matrix_of_bout_durations.append([sim,bout,bee,environment.list_of_bout_durations[bout][bee]])

				# Resources in nest
				matrix_of_resources_foraged.append([sim, bout, bee, environment.quantity_of_nectar_per_bout[bout][bee]])

				if len(environment.list_of_visit_sequences[bout][bee]) > maximum_length_of_a_route : 
					maximum_length_of_a_route = len(environment.list_of_visit_sequences[bout][bee])

		for timestep in range (number_of_timesteps_per_sim) : 
			time = number_of_seconds_per_timesteps*timestep
			for bee in range (number_of_bees) : 
				# Positions
				matrix_of_positions.append([sim, timestep, time, bee, environment.positions_at_each_timestep[timestep,bee]])
				#Coordinates
				#matrix_of_coordinates.append([sim, timestep, time, bee, environment.coordinates_at_each_timestep[timestep,bee]])

		# Arrivals:
		for i in range (len(environment.list_arrival_times)) : 
			matrix_of_arrival_times.append([sim]+environment.list_arrival_times[i])

	# Formatting matrix of visitation sequences to have the same number of columns
	length_of_matrix = len(matrix_of_visit_sequences)       
	for row in range (length_of_matrix) :
		length_of_row = len(matrix_of_visit_sequences[row])
		if length_of_row < maximum_length_of_a_route + 3 : 
			number_of_missing_columns = maximum_length_of_a_route + 3 - length_of_row
			matrix_of_visit_sequences[row] = matrix_of_visit_sequences[row] + [-1 for col in range (number_of_missing_columns)]
	matrix_of_visit_sequences = np.array(matrix_of_visit_sequences)


	# Save the outputs
	np.savetxt(output_folder_of_sim+"\\matrix_of_visit_sequences.csv",matrix_of_visit_sequences, delimiter=',',fmt='%i')

	route_similarity_1_dataframe = pd.DataFrame(matrix_of_similarity_1,columns=["simulation","bout","bee","similarity"])
	route_similarity_1_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\similarity_1_DF.csv', index = False)

	route_similarity_2_dataframe = pd.DataFrame(matrix_of_similarity_2,columns=["simulation","bout","bee","similarity"])
	route_similarity_2_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\similarity_2_DF.csv', index = False)

	route_similarity_3_dataframe = pd.DataFrame(matrix_of_similarity_3,columns=["simulation","bout","bee","similarity"])
	route_similarity_3_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\similarity_3_DF.csv', index = False)

	interactions_dataframe = pd.DataFrame(matrix_of_interactions,columns=["simulation","bout","bee","interactions"])
	interactions_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\interactions_DF.csv', index = False)

	bout_durations_dataframe = pd.DataFrame(matrix_of_bout_durations,columns = ["simulation","bout","bee","bout_duration"])
	bout_durations_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\bout_durations_DF.csv',index=False)

	positions_dataframe  =  pd.DataFrame(matrix_of_positions,columns=["simulation","timestep","time","bee","position"])
	positions_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\positions_DF.csv', index = False)

	quantity_of_nectar_dataframe =  pd.DataFrame(matrix_of_resources_foraged,columns=["simulation","bout","bee","quantity_of_nectar"])
	quantity_of_nectar_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\quantity_of_nectar_DF.csv', index = False)

	arrival_dataframe =  pd.DataFrame(matrix_of_arrival_times,columns=["simulation","timestep","time","bee","flower"])
	arrival_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\arrival_times_DF.csv', index = False)

	return()