'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''
# Parameters --------------------------------------------------------------


########## CAREFUL: HOW TO INITIALIZE THE PARAMETERS ##########
'''
This code allows for testing all combinations of parameters.

The general architecture of each parameter value is either:
	1. parameter = value, and this value will be used for each bee
	2. parameter = [value_for_test_1, ..., value_for_test_m], and m values will be tested in seperate simulations, each bee having the same value for this parameter. Please note that parameter = value and parameter = [value] is equivalent
	3. parameter = [list_of_value_for_each_bee_for_test_1, ..., list_of_value_for_each_bee_for_test_m], which is the same as in case 2 but here, each bee can have different values. The list of different values for each bee must IMPERATIVELY have the a length equal to number_of_bees.
Note that you can combine case 2 and 3 by specifying either a list or a single value for each test.

Let us take the example of alpha, which is used during Q-learning to update the Q tables.
	1. The parameter can take a single value: e.g. alpha_pos = 0.5
	2. The parameter can take a list of values: e.g. alpha_pos = [0.2, 0.5, 0.7]
	3. The parameter can take a list of lists of values: e.g alpha_pos = [[0.2,0.5], [0.5, 0.2], [0.7, 0.5]]

Remark: if you want to test 1 set of values for different bees (only 1 test but specify the value for each bee), do your_parameter = [your_list_of_values]. For example, alpha_pos = [[0.2,0.5]]
'''



########## PARAMETERS TO ADAPT TO THE SIMULATION ##########

# 1. Output processing parameters
experiment_name = "test" # An identification name for the experiment.

# 2. Environment generation parameters
number_of_flowers = 25  # Number of flowers (all patches combined)
number_of_patches = 5 # Number of patches in which the flowers are distributed
density_of_patches = 0 # Takes positive values. 0 corresponds to a uniform spatial distribution

# 3. Bees parameters
number_of_bees = 1 # Number of bees moving simultaneously in the environment.
time_flower_replenishment = 1500 # Time to replenish in seconds
working_memory_span = 1500 # in sec. Time during which a bee is not going to come back to a previously visited flower
alpha = 0.5 # Learning rate
beta = 30  # Exploration-exploitation parameter: 0<=beta




########## PARAMETERS THAT STAY CONSTANT FOR ALL SIMULATIONS ##########

# 1. Output processing parameters
environment_type = "generate"# Either (i) input an array name (refer to folder names in the "Arrays" folder - Note that this folder is automatically created when you first generate an array), or (ii) input "generate" to generate procedural arrays
reuse_generated_arrays =  True # If True and there already are generated arrays with the same parameters, they will be used instead of generating new ones.

# 2. Environment generation parameters
foraging_range = 1000 # Size of the environment in meters. This is half the side of the square. 
number_of_arrays = 25 # Number of different arrays created using the values above. Only used if environment_type == "generate".

# 3. General simulation parameters
duration_of_simulation = 5000*5 # Duration of simulation in seconds
number_of_simulations = 50 # Number of simulations for each set of parameter.
number_of_seconds_per_timesteps = 5 # Duration of a timestep in seconds
create_video_output = False # If  True, outputs videos showing the bee movements during simulation
number_of_simulations_to_show = 1 # Number of simulations to show on video
create_array_visualization = True # If True, will create a visual representation of the array showing the coordinates of each flower and the nest in a 2D space

# 4. Flowers parameters
capacity_of_flower = 10. # Maximal quantity of nectar in one flower in uL

# 4. Bees parameters
dist_factor = 2 # This variable contains the power at which the distance is taken in the [probability = 1/d^distFactor] function to estimate movement probabilities in Q learning
max_distance_travelled = 3000 # Maximum cumulative distance that a bee can fly before going home
max_crop_capacity = 50. # Volume of bee crop in uL
velocity = 5 # Flight velocity of bees in meters per second
time_inactive_in_nest = 440 # Duration of rest in nest in seconds after each bout
foraging_rate = 1 # Rate of depletion of nectar in flowers, in uL/s
perception_range = 5 # If allowing_discovery, radius of perception of flowers around the bee when she discovers new flowers, in meters

# 5. Extra model features that are not used for these simulations
allowing_discovery = False # If False will disable discovery of flowers and assume the bees already know every flowers position at the beginning of the simulation
cost_of_discovery = 5 # If allowing_discovery, multiplicative factor that increases the distance flown by the bees when discovering a new flower. Put 1 if don't want any cost
allowing_memory_decay = False # If False, will assume that the bees will remember the flower locations and quality forever and perfectly
memory_span = 4000*5 # If allowing_memory_decay, Duration in seconds during which a site will be kept in memory
force_full_crop = False # Force the bees to forage until their crop is full (ignore max_distance_travelled ans stochastic chances to choose the nest as next destination)
proba_discovery_when_max_crop = 0.5 # If allowing_discovery, when memory = max number of full flowers in crop, this is the probability of discovering new flowers
gamma = 0  # In Q-learning, teporal discounting factor: 0<=gamma<=1. If >0, it assumes that bees take into account further steps in their decision-making. Here, set to 0 for parsimony
probability_of_recruitment = 0. # Probability that bees are sharing the location of the best flower found during foraing with a nestmate. Simulates waggle dace in honeybees. Set to 0 if no recruitment
social_learning_rate = 0. # If probability_of_recruitment>0, rate at which rate the bees are learning social information




########## CONVERT TIMES IN TIMESTEPS ##########
number_of_timesteps_per_sim = int(duration_of_simulation/number_of_seconds_per_timesteps) # Number of timesteps in one simulation
timesteps_inactive_in_nest = int(time_inactive_in_nest/number_of_seconds_per_timesteps) # Number of timesteps the bee is resting in nest
quantity_nectar_eaten_per_timestep = foraging_rate*number_of_seconds_per_timesteps


########## WARNING ##########
minimal_recommended_duration = (max_distance_travelled + foraging_range)/(velocity)
if duration_of_simulation < minimal_recommended_duration : 
	print("Careful: the duration of the simulation might not be enough to generate at least one bout for each bee. \nWe would recommend to use at least "+ 
		str(minimal_recommended_duration)+" seconds. \nThis parameter can be changed in parameters.py\n")


# If environment_type is not a "generate", there is no need for multiple arrays.
if(environment_type!="generate") : 
	number_of_arrays = 1;




########## INITIALIZING THE PARAMETERS DICTIONARIES ##########

global_array_info = {
"environment_type": environment_type, 
"foraging_range": foraging_range,
"number_of_flowers" :number_of_flowers,
"number_of_patches": number_of_patches,
"density_of_patches": density_of_patches,
"perception_range" : perception_range, 
}

for key in global_array_info : 
  if not isinstance(global_array_info[key],list) : 
  	global_array_info[key] = [global_array_info[key]]

simulation_parameters = {
"number_of_bees" : number_of_bees,
"time_flower_replenishment" : time_flower_replenishment,
"dist_factor" : dist_factor,
"allowing_discovery": allowing_discovery,
"allowing_memory_decay": allowing_memory_decay,
"memory_span": memory_span,
"working_memory_span":working_memory_span,
"proba_discovery_when_max_crop" : proba_discovery_when_max_crop,
"cost_of_discovery": cost_of_discovery,
"force_full_crop":force_full_crop
}

parameters_of_individuals  = {
"max_distance_travelled": max_distance_travelled,
"max_crop_capacity":max_crop_capacity,
"perception_range":perception_range,
"velocity": velocity,
"timesteps_inactive_in_nest":timesteps_inactive_in_nest,
"quantity_nectar_eaten_per_timestep":quantity_nectar_eaten_per_timestep,
"beta": beta,
"alpha": alpha,
"gamma":gamma,
"probability_of_recruitment" : probability_of_recruitment,
"social_learning_rate" : social_learning_rate
}

parameters_loop = {}
parameters_loop.update(parameters_of_individuals)
parameters_loop.update(simulation_parameters)

for key in parameters_loop : 
  if not isinstance(parameters_loop[key],list) : 
  	parameters_loop[key] = [parameters_loop[key]]

time_constants = {
"number_of_seconds_per_timesteps" : number_of_seconds_per_timesteps,
"number_of_timesteps_per_sim" : number_of_timesteps_per_sim,
"duration_of_simulation" : duration_of_simulation,
"capacity_of_flower" : capacity_of_flower
}