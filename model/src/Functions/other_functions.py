'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Other Functions  --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
Set of secondary functions to help with the simulations.
'''

import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sns
import pandas as pd 
import numpy as np 
import imageio
import os
import warnings

def softmax(values_vector,beta) : 
    
    # Returns the probabilities of choosing options characterised by some values by using a softmax decision function

    if len(values_vector) == 0 : 
      return([])
    else : 
      values_vector = np.array(values_vector)
      

      with warnings.catch_warnings():
          warnings.filterwarnings('error')
          try:
              p = np.exp(beta*values_vector)/np.sum(np.exp(beta*values_vector))
          except Warning as e:
              p = (values_vector==np.max(values_vector)).astype(float)
      return(p)

def character_match(str_list,chr_searched) : 
  
  # Returns a list chr_matched such as chr_matched[i]=True if chr_searched is found in the string str_list[i] and chr_matched[i]=False otherwise

  if len(str_list) == 0 : 
    return(False)
  else : 
    chr_matched = np.array([chr_searched in str_list[k] for k in range (len(str_list))])
    return(np.array(chr_matched))

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_video_output(gif_name,array_geometry,coordinates_at_each_timestep,number_of_bees,time_constants,output_folder_of_sim,
  evolution_of_memory,memory_at_each_timestep, from_timestep = 0) : 

  # Creates a GIF file to show bees movements and their memory change through time (starting at timetep from_timestep)

  number_of_timesteps_per_sim = time_constants["number_of_timesteps_per_sim"]
  number_of_seconds_per_timesteps = time_constants["number_of_seconds_per_timesteps"]

  # Make all the frames
  print('\nPlotting the frames of the output video')

  list_of_frames_paths = []

  previous_pos = np.full((number_of_bees,2),None)
  current_pos = coordinates_at_each_timestep[0,:,:]

  colors= cm.viridis(np.linspace(0, 1, number_of_bees))

  lines = [[] for bee in range (number_of_bees)]

  # Set size of dots
  if evolution_of_memory : memory_at_each_timestep =  memory_at_each_timestep/np.max(memory_at_each_timestep)*2000

  fig,ax=plt.subplots(figsize=(8,8))
  sns.scatterplot(data=array_geometry, x='x',y='y',style='patch',color='black',s=100,alpha=0.75, legend=None)

  annotation = None

  print_progress_bar(0, number_of_timesteps_per_sim-from_timestep, prefix = 'Progress:', suffix = 'Complete', length = 50)

  for timestep in range (from_timestep, number_of_timesteps_per_sim) : 

    if timestep == from_timestep :

      if evolution_of_memory: 
        scatter = []
        for bee in range (number_of_bees) : 
          scatter.append (ax.scatter(array_geometry.loc[:,"x"],array_geometry.loc[:,"y"],
            s=memory_at_each_timestep[timestep*number_of_bees+bee,:],alpha=0.25,color=colors[bee]))

      for bee in range (number_of_bees) : 
        ax.scatter(current_pos[bee,0],current_pos[bee,1],marker='.',s=100,color=colors[bee],label='bee '+str(bee))

    else : 

      if evolution_of_memory : 
        for bee in range (number_of_bees) :
          scatterplot = scatter[bee]
          scatterplot.set_visible(False)
          scatter[bee] = ax.scatter(array_geometry.loc[:,"x"],array_geometry.loc[:,"y"],
            s=memory_at_each_timestep[timestep*number_of_bees+bee,:],alpha=0.25,color=colors[bee])

      for bee in range (number_of_bees) : 

        if np.array_equal(previous_pos[bee,:],np.zeros(2)) and np.array_equal(current_pos[bee,:],np.zeros(2)) :
          for line in lines[bee] :
            line = line.pop(0)
            line.remove()
          lines[bee]=[]

        else : 
          line  = ax.plot([previous_pos[bee,0],current_pos[bee,0]],[previous_pos[bee,1],current_pos[bee,1]],color=colors[bee])
          lines[bee].append(line)

      ax.texts.remove(annotation)

    annotation = ax.annotate("time = "+str(number_of_seconds_per_timesteps*timestep)+" sec.",xy=(0,1.05), xycoords=ax.get_xaxis_transform())


    ax.axis('equal')
    plt.legend()

    frame_path = output_folder_of_sim+"\\frame_"+str(timestep)+".png"
    list_of_frames_paths.append(frame_path)

    plt.savefig(frame_path)

    if timestep != number_of_timesteps_per_sim - 1 :
      previous_pos = current_pos
      current_pos = coordinates_at_each_timestep[timestep+1,:,:]

    print_progress_bar(timestep-from_timestep+1, number_of_timesteps_per_sim-from_timestep, prefix = 'Progress:', suffix = 'Complete', length = 50)

  plt.close()


  # Make the gif
  print('Making the gif')

  gif_path = output_folder_of_sim+'\\'+gif_name

  print_progress_bar(0, number_of_timesteps_per_sim-from_timestep, prefix = 'Progress:', suffix = 'Complete', length = 50)

  with imageio.get_writer(gif_path, mode='I') as writer:
      for timestep in range (from_timestep, number_of_timesteps_per_sim) : 
        path = list_of_frames_paths[timestep-from_timestep]
        frame = imageio.imread(path)
        writer.append_data(frame)
        print_progress_bar(timestep-from_timestep+1, number_of_timesteps_per_sim-from_timestep, prefix = 'Progress:', suffix = 'Complete', length = 50)

  # Remove files
  print('Removing the files\n')

  for path in set(list_of_frames_paths):
      os.remove(path)