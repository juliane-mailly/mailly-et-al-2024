'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Management of data functions  --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
Set of fonctions that allows to manage the input and output data of the simulations.
'''

import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy
import geometry_functions
import environment_generation_functions


def get_list_of_parameters_names(parameters_loop):

  # Gets the name of the parameters in the parameters dictionary

  return(list(parameters_loop.keys()))

def initialize_probability_matrix(array_geometry,dist_factor) : 

  # Initialises the probability of moving from one flower to the next

  distance_between_flowers = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)
  initial_probability_matrix = geometry_functions.normalize_matrix_by_row (geometry_functions.give_probability_of_vector_with_dist_factor(distance_between_flowers,
    dist_factor))

  return (np.array(initial_probability_matrix))

def initialize_environment_matrices(dist_factor,bee_info,array_geometry,number_of_seconds_per_timesteps) : 

  # Initialises a dictionary of matrices each containing information about the environment

  velocity = bee_info["velocity"][0]
  distance_between_flowers = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)
  timesteps_between_flowers = np.ceil(distance_between_flowers/(velocity*number_of_seconds_per_timesteps))
  probabilities_discovering_flowers = initialize_probability_matrix(array_geometry,dist_factor)
  matrix_of_flowers_perceived = geometry_functions.get_matrix_of_flowers_perceived(array_geometry, bee_info.loc[0,"perception_range"])
  non_norm_prob = geometry_functions.give_probability_of_vector_with_dist_factor(distance_between_flowers,dist_factor)

  environment_matrices = {"timesteps_between_flowers":timesteps_between_flowers,
  "probabilities_discovering_flowers":probabilities_discovering_flowers,
  "matrix_of_flowers_perceived":matrix_of_flowers_perceived,
  "distance_between_flowers":distance_between_flowers,
  "non_norm_prob":non_norm_prob}

  return(environment_matrices)

def initialize_Q_table(allowing_discovery, array_geometry, time_constants, environment_matrices) : 

  # Initialises the Q-values
  number_of_states = len(array_geometry.index)

  if not allowing_discovery : 
    Q_table = (time_constants["capacity_of_flower"]/2)*environment_matrices["probabilities_discovering_flowers"] 
    return(Q_table)

  else :
    Q_table = np.zeros((number_of_states,number_of_states))
    Q_table[:, 0] = (time_constants["capacity_of_flower"]/2)*environment_matrices["probabilities_discovering_flowers"][:,0]
    return(Q_table)

def initialize_bee_info(parameters_dict) :

  # Initialises the dictionary of individual bees parameters

  number_of_bees = parameters_dict["number_of_bees"]

  for key in parameters_dict : 
    if not isinstance(parameters_dict[key],list) :
      parameters_dict[key] = [parameters_dict[key] for k in range (number_of_bees)] 
    else : 
      if len(parameters_dict[key]) != number_of_bees : 
        raise ValueError("Impossible to initialize dataframe of bee info because of parameter "+str(key)+
          " contains lists whose number of elements is different from the number of bees.\nPlease refer to parameter.py for a full description of the initialization of the parameters")  
  dict_of_bee_info = {"ID": [bee for bee in range (number_of_bees)]}
  dict_of_bee_info.update(parameters_dict)
  bee_info = pd.DataFrame(dict_of_bee_info)
  return(bee_info)

def add_array_ID_to_bee_info(bee_info, array_ID,number_of_bees) :

  # Adds the current array's ID to the bee_info dictionary

  bee_info["array_ID"] = [array_ID for k in range (number_of_bees)]
  return()

def make_arrays_and_output_folders() : 

  # Manages the folders

  try : 
    os.mkdir("Output")
  except : 
    pass 
  try : 
    os.mkdir("Arrays")
  except : 
    pass

def create_name_of_test(environment_type,experiment_name,index_of_parameter_set) :

  # Gives a unique test ID

  time_now = (datetime.now()).strftime("%Y%m%d%Y%H%M%S")
  test_name = environment_type+"-"+experiment_name+"-param_set"+str(index_of_parameter_set+1)+"-"+time_now
  return(test_name)

def initialize_all_environments(number_of_arrays, array_info, reuse_generated_arrays, current_working_directory, create_array_visualization) :
  
  # Initialises the arrays

  list_of_arrays = []

  for array_number in range (number_of_arrays) : 

    # Initialize array
    array_geometry, array_info, array_folder = environment_generation_functions.create_environment(array_info, array_number, reuse_generated_arrays, 
      current_working_directory, create_array_visualization)

    array_info_bis = copy.deepcopy(array_info)
    list_of_arrays = list_of_arrays + [(array_geometry, array_info_bis, array_folder)]
  
  return(list_of_arrays)

def initialize_data_of_current_test(list_of_names_of_parameters,parameter_values,environment_type,experiment_name,index_of_parameter_set,
  current_working_directory,time_constants,seed):

  # Create test name according to parameter values
  test_name = create_name_of_test(environment_type,experiment_name,index_of_parameter_set)

  # Create the output folder for this test in the Output directory
  output_folder_of_test = current_working_directory + "\\Output\\"+ test_name
  os.mkdir(output_folder_of_test)

  # Complete list of individual parameters. These are initialized with parameters_loop
  parameters_dict = dict(zip(list_of_names_of_parameters,parameter_values))


  # Create a dataframe of information to be retrieve in further analyses and passed to the simulation functions (and remember what parameters were used).
  bee_info = initialize_bee_info(parameters_dict)
  if bee_info.loc[0,"working_memory_span"]=="matching" : 
    bee_info.loc[:,"working_memory_span"] = bee_info["time_flower_replenishment"]

  # Save the time constants in this folder
  time_constants["time_flower_replenishment"] = bee_info["time_flower_replenishment"][0]
  time_constants_df = {}
  for key in time_constants : 
    time_constants_df[key] = [time_constants[key]]
  time_constants_df = pd.DataFrame(time_constants_df)
  time_constants_df.to_csv(path_or_buf = output_folder_of_test+'\\time_constants.csv',index = False)
  pd.DataFrame({"seed":[seed]}).to_csv(path_or_buf = output_folder_of_test+'\\seed.csv', index = False, header = False)

  return(output_folder_of_test,bee_info)

def initialize_data_of_current_array(array_info, array_number, number_of_bees,bee_info,output_folder_of_test,time_constants, dist_factor,array_geometry):

  number_of_seconds_per_timesteps = time_constants ["number_of_seconds_per_timesteps"]

  # Initilialize time/space constants
  environment_matrices = initialize_environment_matrices(dist_factor,bee_info,array_geometry,number_of_seconds_per_timesteps)

  # Initialize learning array list
  initial_Q_table = initialize_Q_table(bee_info["allowing_discovery"][0], array_geometry, time_constants, environment_matrices)
  
  # Add array_ID to bee_info
  add_array_ID_to_bee_info(bee_info, array_info["array_ID"],number_of_bees)

  # Create an Array folder for the current array in the Output folder
  output_folder_of_sim = output_folder_of_test + "\\Array"+"{:02d}".format(array_number)
  os.mkdir(output_folder_of_sim)

  # Save bee_info
  pd.DataFrame(bee_info).to_csv(path_or_buf = output_folder_of_sim+'\\bee_info.csv', index = False)

  return(output_folder_of_sim, initial_Q_table, environment_matrices)