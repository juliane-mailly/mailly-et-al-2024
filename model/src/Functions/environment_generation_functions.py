'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Environment generation functions --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
Set of functions to generate environments or retrieve information on previously generated environments.
'''

import os
import numpy as np
import pandas as pd
import re
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import geometry_functions
import other_functions

def normalize_array_geometry(array_geometry) : 
  
  # Sets the first line (nest) coordinates' at (0,0) and adjust all other coordinates.

  array_geometry["x"] = array_geometry["x"] - array_geometry["x"][0]
  array_geometry["y"] = array_geometry["y"] - array_geometry["y"][0]
  return(array_geometry)


def load_and_normalize_array_geometry(file_name) : 

  # Loads an array_geometry.csv file and normalize it

  return(normalize_array_geometry(pd.read_csv(file_name)))


def create_array_ID(array_info,array_number) : 
  
  # Create an unique name for an array

  array_ID = "Array-" + str(array_info["number_of_flowers"]) + "-"+ str(array_info["number_of_patches"]) + "-" + str(array_info["density_of_patches"]) +"-" + str(array_info["foraging_range"]) + "_" + "{:02d}".format(array_number)
  return(array_ID)


def get_ID_number_for_array(list_of_known_arrays,array_type_ID):
  
  #Scans among known arrays to find a array ID not taken to give to the new array

  indices_of_similar_arrays = np.where(other_functions.character_match(list_of_known_arrays,array_type_ID))[0]
  list_of_similar_arrays = list_of_known_arrays[indices_of_similar_arrays]
  number_of_similar_arrays = len(list_of_similar_arrays)
  available_numbers = [1 for i in range (number_of_similar_arrays+1)] # available_number[number] = 1 if number is abailable and 0 otherwise
  if number_of_similar_arrays > 0 :
    for i in range (number_of_similar_arrays) : 
      name_of_scanned_array = list_of_similar_arrays[i] 
      end_of_scanned_array = re.findall('\_0*\d*',name_of_scanned_array)[0]
      scanned_array_number = int(end_of_scanned_array[1:])
      available_numbers[scanned_array_number] = 0 # this number is not available
  array_ID_number = np.where(np.array(available_numbers)==1)[0][0]  # smallest number such as available_numbers[number] == 1 (and add 1 because we want the ID to start from 1)
  return(array_ID_number)


def generate_cartesian_coordinates_for_flower(min_distance,max_distance) :  # can be used for patch centers
  
  # Generates cartesian coordinates for a flower or the center of a patch

  distance = np.random.uniform(low=min_distance,high=max_distance)
  azimuth = np.random.uniform(low=0,high=2*np.pi)
  return(geometry_functions.convert_polar_to_cartesian((distance,azimuth)))


def check_if_patch_is_sufficiently_far(coordinates_of_current_patch, minimal_distance_between_patches ,coordinates_of_other_patches) : # can be used for path centers
  
  # Checks if a flower/patch is at a sufficient distance from orther flowers/patches

  number_of_patches_to_check, _ = np.shape(coordinates_of_other_patches)
  distances_with_other_patches = [geometry_functions.distance(coordinates_of_current_patch,coordinates_of_other_patches[flower]) for flower in range(number_of_patches_to_check)]

  return (np.min(distances_with_other_patches)>=minimal_distance_between_patches)



def generate_array_procedurally(number_of_flowers,number_of_patches,density_of_patches,foraging_range,perception_range) : 

    # Generate an environment of flowers procedurally.


  #perception_range = 10 # Use to be a parameter, but no longer used. As it is needed for this function, it is set as a constant here. Any positive value is fine.

  if density_of_patches < 0 : 
    raise ValueError("Unsupported density of patches. Values must be positive.")
  if number_of_patches > number_of_flowers : 
    raise ValueError("You must have at least one flower per patch.")
  if number_of_flowers == 0 or number_of_patches == 0 :
    raise ValueError("You must at least have one flower and one patch.")
  if (4*perception_range >= foraging_range) :
    raise ValueError("The environment size must be at least superior to 4 times the perception range.")

  MARGIN_OF_ENVIRONMENT = 10
  MINIMAL_DISTANCE_PATCH_PATCH = 2*perception_range
  MAX_TRIALS_TO_PLACE_CENTER = 200
  INCREASE_ENVIRONMENT_SIZE = foraging_range/20


  # Special case if density_of_patches == 0: uniform environment. We create as much patches as there are flowers in the environment
  if density_of_patches == 0:
    number_of_patches = number_of_flowers

  # Draw the coordinates of the centers of the patches and check if they respect the minimal distance between patches (the nest is considered as a patch center here)   

  patch_centers_coordinates = [np.array([0,0])]

  while(len(patch_centers_coordinates)!=number_of_patches+1) :
    coordinates_of_current_patch = np.random.uniform(low = -foraging_range+MARGIN_OF_ENVIRONMENT, high = foraging_range-MARGIN_OF_ENVIRONMENT, size=2)   
    patch_center_far_enough = check_if_patch_is_sufficiently_far(coordinates_of_current_patch, MINIMAL_DISTANCE_PATCH_PATCH, patch_centers_coordinates) 
    trial = 0

    while (not patch_center_far_enough) and trial<MAX_TRIALS_TO_PLACE_CENTER : 
      coordinates_of_current_patch = np.random.uniform(low = -foraging_range+MARGIN_OF_ENVIRONMENT, high = foraging_range-MARGIN_OF_ENVIRONMENT, size=2)   
      patch_center_far_enough = check_if_patch_is_sufficiently_far(coordinates_of_current_patch, MINIMAL_DISTANCE_PATCH_PATCH, patch_centers_coordinates)
      trial += 1

    if patch_center_far_enough : 
      patch_centers_coordinates.append(coordinates_of_current_patch)

    else : 
      foraging_range += INCREASE_ENVIRONMENT_SIZE

  # If density_of_patches == 0: the patch centers will be the flowers
  if density_of_patches == 0:
    patch_centers_coordinates = np.array(patch_centers_coordinates)
    array_geometry = pd.DataFrame({"ID":[flower for flower in range (number_of_flowers+1)],"x":patch_centers_coordinates[:,0],"y":patch_centers_coordinates[:,1],"patch":[flower for flower in range (number_of_flowers+1)]})

  else : 

    # For each flower, decide the patch to which it will belong (patch 0 is the nest so there will be no flower assigned to it)
    flowers_ID_coordinates_patch = [np.array([0,0,0,0])] # Contains the future output (flower ID, x coordinate, y coordinate, patch ID). The nest is considered a flower (in patch 0)
    flowers_belonging_to_patch = np.random.randint(low = 1, high = (number_of_patches+1), size = number_of_flowers)
    number_of_flower_per_patch = np.bincount(flowers_belonging_to_patch,minlength=number_of_patches+1)

    flower_ID = 1

    for patch in range (number_of_patches+1) :
      
      radius_of_patch = np.sqrt(number_of_flower_per_patch[patch]/(density_of_patches*np.pi))

      for flower in range(number_of_flower_per_patch[patch]) : 

        x,y = generate_cartesian_coordinates_for_flower(0,radius_of_patch)
        coordinates_of_flower = patch_centers_coordinates[patch]+np.array([x,y])
        flowers_ID_coordinates_patch.append(np.concatenate(([flower_ID],coordinates_of_flower,[patch])))
        flower_ID += 1

    array_geometry = pd.DataFrame(np.array(flowers_ID_coordinates_patch),columns = ["ID","x","y","patch"])

  return(array_geometry)



def generate_and_save_visual_representation_array(array_geometry, array_info, array_folder):
  
  # Creates a visual representation of the array showing the coordinates of each flower and the nest in a 2D space. If such file already exist, will not do anything
  

  known_files =  np.array(os.listdir(array_folder))
  name_of_file = "array_visualization.png"
  if not name_of_file in known_files : 
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    sns.scatterplot(data=array_geometry, x='x',y='y',hue='patch',palette='viridis',s=100,ax=ax)
    ax.axis('equal')
    plt.title(array_info['array_ID'])
    path = array_folder + '\\array_visualization.png'
    plt.savefig(path)
    plt.close()
  return()



def create_environment (array_info, array_number, reuse_generated_arrays,current_working_directory, create_array_visualization) : 

  # Places the nest and flowers in a spatial plane according to the inputs given.


  # This code is used if you generate the environment procedurally
  if (array_info['environment_type'] == "generate"): 

    # Create a name for the folder of the array
    array_info['array_ID'] = create_array_ID(array_info,array_number) 

    # Retrieve the list of arrays available in the Arrays folder
    list_of_known_arrays = np.array(os.listdir(current_working_directory + '\\Arrays'))
    array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']

    # If reuse_generated_arrays, look for similar arrays
    found_similar_array = False 
    if (reuse_generated_arrays) : 
      indices_similar_array = np.where(other_functions.character_match(list_of_known_arrays,array_info['array_ID']))[0]
      found_similar_array = (len(indices_similar_array)!=0)
      if found_similar_array : 
        array_geometry = pd.read_csv(array_folder+'\\array_geometry.csv')

    # If not reuse_generated_arrays or there was no similar arrays, generate a new one
    # Adjust arrayID to make sure there is no overlap with an existing array
    if not reuse_generated_arrays or not found_similar_array : 
      index_last_char_array_type_ID = re.search('\_0*\d*',array_info['array_ID']).span()[0]
      array_type_ID = array_info['array_ID'][:index_last_char_array_type_ID]
      array_file_number = get_ID_number_for_array (list_of_known_arrays,array_type_ID)
      array_info['array_ID'] = create_array_ID(array_info,array_file_number)
      array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']



      os.mkdir(array_folder)

      array_geometry = generate_array_procedurally(array_info['number_of_flowers'],array_info['number_of_patches'],
          array_info['density_of_patches'],array_info['foraging_range'],array_info["perception_range"])

      # Write the array and parameters
      array_info_saved = copy.deepcopy(array_info)
      for key in array_info_saved : 
        array_info_saved[key] = [array_info_saved[key]]
        
      pd.DataFrame(array_info_saved).to_csv(path_or_buf = array_folder + '\\array_info.csv', index = False)
      array_geometry.to_csv(path_or_buf = array_folder + "\\array_geometry.csv", index = False)

  # This code is used if you want to use a specific array
  else :
    array_folder = current_working_directory+"\\Arrays\\"+array_info["environment_type"]
    array_geometry = load_and_normalize_array_geometry(array_folder+"\\array_geometry.csv")
    array_info = pd.read_csv(array_folder+"\\array_info.csv").to_dict(orient='list')
    for key in array_info : 
      array_info[key] = array_info[key][0]
    array_info['array_ID'] = array_info['environment_type']

  # If create_array_visualization, will create a plot showing the array (if not already here)
  if create_array_visualization : 
    generate_and_save_visual_representation_array(array_geometry, array_info, array_folder)

  return(array_geometry, array_info, array_folder)