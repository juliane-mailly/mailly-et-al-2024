'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Main --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
This piece of code uses mutiprocessing to parallelise the execution of the different groups of simulations. 
Simulations with different bee parameters are traeted in parallel with the loop function.
If different environment parameters combinations are specified, they will be executed one after the other to avoid environent generation conflicts.
'''


import multiprocessing as mp
import numpy as np
import time
import itertools
import os
import random


def loop(parameter_values, index_of_parameter_set, list_of_names_of_parameters, environment_type, time_constants, experiment_name, 
	current_working_directory,number_of_simulations,create_video_output,number_of_simulations_to_show,
	simulation_parameters,list_of_arrays,seed):

	# Main loop for parallelising each bee parameters' simulations.

	import sys
	sys.path.append('Functions')
	import management_of_data_functions
	import simulation_functions

	np.random.seed(seed)
	random.seed(seed)


	parameter_values = list(parameter_values)

	# Initializing -------------------------------------------------------------------------------------------------
	
	# Initialize data of current test
	output_folder_of_test, bee_info = management_of_data_functions.initialize_data_of_current_test(list_of_names_of_parameters,
		parameter_values,environment_type,experiment_name,index_of_parameter_set,current_working_directory,time_constants, seed)
	
	# Loop on the different arrays
	for array_number,array in enumerate(list_of_arrays) : 
		
		# Retrieve array information
		array_geometry, array_info, array_folder = array

		# Initialize data of array (also dependent on the bees parameters, that is why it is computed now)
		output_folder_of_sim, initial_Q_table, environment_matrices = management_of_data_functions.initialize_data_of_current_array(array_info, 
			array_number, bee_info.loc[0,"number_of_bees"], bee_info, output_folder_of_test,time_constants, simulation_parameters["dist_factor"],array_geometry)

		# Run the simulations 
		simulation_functions.run_all_simulations_and_make_output_files(initial_Q_table,environment_matrices,number_of_simulations,
			time_constants,bee_info.loc[0,"number_of_bees"],bee_info,array_geometry,output_folder_of_sim,create_video_output,number_of_simulations_to_show)


	return()







if __name__ == '__main__':

	# Main piece of code that loops over the different combinations of environment parameters, then parallelises the execution of te simulations for each set of bee parameters.

	with mp.Pool(mp.cpu_count()) as p:

		import os
		current_working_directory = os.getcwd()

		import sys
		sys.path.append('Functions')

		from parameters import *
		import management_of_data_functions


		# Create Output directory in the current working directory.
		management_of_data_functions.make_arrays_and_output_folders()

		# Make and save the seed for he following simulations
		seed = np.random.randint(0,1000)


		# Decompose global_array_info into array_info dicts containing the right information
		list_array_param_names = management_of_data_functions.get_list_of_parameters_names(global_array_info)
		list_array_param_sets = list(itertools.product(*list(global_array_info.values())))

		# Get starting time of simulation to get computation time.
		start_of_simulation = time.time()
	
		# Loop over all the array_info dicts
		for param_set in list_array_param_sets : 
			array_info = dict(zip(list_array_param_names, param_set))

			# Initialize all environments before starting parallelization
			list_of_arrays = management_of_data_functions.initialize_all_environments(number_of_arrays, array_info, reuse_generated_arrays, current_working_directory, 
				create_array_visualization)

			# Successive loops for each parameter. Thus, all parameter combinations are tested. This code allows to add new parameters in parameters_loop without changin the loop code
			list_of_names_of_parameters = management_of_data_functions.get_list_of_parameters_names(parameters_loop)

			list_of_parameter_sets = list(itertools.product(*[parameters_loop[param] for param in list_of_names_of_parameters]))
			total_number_of_parameter_sets = len(list_of_parameter_sets)

			# Loop on the possible sets of parameters
			p.starmap(loop, [(list_of_parameter_sets[index_of_parameter_set], index_of_parameter_set, list_of_names_of_parameters, array_info["environment_type"], 
				time_constants, experiment_name, current_working_directory, number_of_simulations,
				create_video_output,number_of_simulations_to_show,simulation_parameters,list_of_arrays,seed) for 
				index_of_parameter_set in range (total_number_of_parameter_sets)])

		end_of_simulation = time.time()
		duration_simulation = end_of_simulation - start_of_simulation
			
		print("Simulation completed in "+str(round(duration_simulation,5))+" seconds.")

	p.close()
