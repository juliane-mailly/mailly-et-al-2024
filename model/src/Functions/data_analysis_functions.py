'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Data analysis functions --------------------------------------------------------------

########## STRUCTURE OF THE CODE ##########
'''
Set of functions to analyse data, either when the code is running or afterwards
'''


import pandas as pd 
import numpy as np 
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


################################# OVERLAP ROUTES #################################

def compute_overlap(array_info_to_save, list_of_parameter_to_save,array_number,matrix_of_visit_seq) : 

  nrow = len(matrix_of_visit_seq.index)
  output = []

  nsim = np.max(matrix_of_visit_seq[0])
  nbee = np.max(matrix_of_visit_seq[2])

  row = 0
  
  for sim in range (nsim+1) : 
    for bee in range (nbee+1) : 


      data = matrix_of_visit_seq[(matrix_of_visit_seq[0] == sim)&(matrix_of_visit_seq[2] == bee)].sort_values(1)
      nbouts = len(data)

      for bout in range(nbouts - 1) : 


        visit_seq_1 = np.array(data.iloc[bout,3:])
        visit_seq_1 = visit_seq_1[visit_seq_1>0] # remove nest and all the -1  

        visit_seq_2 = np.array(data.iloc[bout+1,3:])
        visit_seq_2 = visit_seq_2[visit_seq_2>0] # remove nest and all the -1     

        overlap = np.sum(np.unique(visit_seq_1) == np.unique(visit_seq_2))    

        output.append(array_info_to_save+list_of_parameter_to_save+[array_number, sim, bout, bee, overlap])

  return(output)


################################# INTERVISIT DURATION #################################

def compute_intervisit_duration(array_info_to_save,list_of_parameter_to_save,array_number,arrival_times_DF,
  number_of_timesteps_in_time_window,number_of_bees,number_of_timesteps_per_sim,number_of_flowers, number_of_seconds_per_timestep) : 

  starting_row = 0
  timestep_min = 0
  timestep_max = number_of_timesteps_in_time_window

  output = []
  sim = 0
  row = 0
  timestep =  arrival_times_DF.loc[row,"timestep"]


  while (timestep_max < arrival_times_DF[arrival_times_DF["simulation"]==sim]["timestep"].max()) : 

    last_visit = np.full((number_of_bees, number_of_flowers),None)
    time_between_visits = [[] for i in range (number_of_bees)]

    while timestep < timestep_max : 

      bee = arrival_times_DF.loc[row,"bee"]
      position = int(arrival_times_DF.loc[row,"flower"])
      time = arrival_times_DF.loc[row,"time"]

      if last_visit[bee, position] is not None : 

        time_between_visits[bee].append(time - last_visit[bee, position])

      last_visit[bee, position] = time

      row += 1
      timestep = arrival_times_DF.loc[row,"timestep"]

    for bee in range (number_of_bees) : 
      if len(time_between_visits[bee]) != 0 : 
        average_intervisit = np.mean(time_between_visits)
        output.append(array_info_to_save+list_of_parameter_to_save+[array_number,sim, timestep_min, timestep_min*number_of_seconds_per_timestep, bee, average_intervisit])


    starting_row = row
    timestep_min += number_of_timesteps_in_time_window
    timestep_max += number_of_timesteps_in_time_window

    if (arrival_times_DF[arrival_times_DF["simulation"]==sim]["timestep"].max() <= timestep_max) : 

      sim = sim+1
      row = arrival_times_DF[arrival_times_DF["simulation"]==sim].index.min()
      starting_row = row
      timestep_min = 0
      timestep_max = number_of_timesteps_in_time_window
      timestep = 0

  return(output)




################################# AVERAGE REPLENISHMENT #################################


def compute_freq_fast(array_info_to_save, list_of_parameter_to_save,array_number,matrix_flower_type,number_of_bees, min_replenishment) : 

  nrow = len(matrix_flower_type.index)
  output = []

  for i in range (nrow) : 

    seq = np.array(matrix_flower_type.iloc[i,3:])
    seq = seq[seq>0] # remove all the -1

    row_of_dataframe = list(matrix_flower_type.iloc[i,:3])

    if len(seq)>0 : freq_fast = np.mean(seq==min_replenishment)
    else : freq_fast = None

    output.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[freq_fast]) 

  return(output)



################################# FORAGING SUCCESS #################################


def compute_foraging_success(array_info_to_save, list_of_parameter_to_save, array_number, bout_durations_DF, quantity_of_nectar_DF,number_of_bees) :

  nrow = len(bout_durations_DF.index)
  output = []

  for i in range (nrow) : 

    foraging_success = quantity_of_nectar_DF.iloc[i,-1]/bout_durations_DF.iloc[i,-1]
    row_of_dataframe = list(bout_durations_DF.iloc[i,:3])    
    output.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[foraging_success]) 

  return(output)

################################# NFLOWERS/LENGTH OF BOUT #################################

def compute_nflowers_dataframe(array_info_to_save, list_of_parameter_to_save,array_number,matrix_of_visit_seq) : 

  nrow = len(matrix_of_visit_seq.index)
  output = []

  for i in range (nrow) : 

    visit_seq = np.array(matrix_of_visit_seq.iloc[i,3:])
    visit_seq = visit_seq[visit_seq>0] # remove nest and all the -1

    row_of_dataframe = list(matrix_of_visit_seq.iloc[i,:3])

    nflowers = len(np.unique(visit_seq))
    length_bout = len(visit_seq)
    
    output.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[nflowers, length_bout]) 

  return(output)



################################# ENTROPY #################################


def conditional_entropy_pth_order(sequence, p, H_0 = None) : 
  
  sequence = np.array(sequence)
  flowers = np.unique(sequence)
  number_of_flowers = len(flowers)

  if p == 0 : 
    occurences = [np.sum(sequence == flower) for flower in flowers]
    probabilities = np.array(occurences)/number_of_flowers
    H_p = entropy(probabilities, base = number_of_flowers)

    H_p_corr = H_p #H_p_corr doesn't really matter here

  else : 


    # Fill the dictionary of transitions
    number_of_subseq = len(sequence)-p
    transitions = dict()

    for i in range(number_of_subseq) :  
      s_p, s_i = tuple(sequence[i:i+p]), sequence[i+p]
      
      if s_p in transitions : 
        if s_i in transitions[s_p] : transitions[s_p][s_i] += 1
        else : transitions[s_p][s_i] = 1 
      
      else :
        transitions[s_p] = {s_i : 1}

      
    # Get the conditional and unconditional probabilities
    
    sum_transitions = {}
    unconditional_probabilities = {}
    conditional_probabilities = {}
    number_subseq_one_occ = 0
    
    for s_p in transitions : 
      
      # Sum transitions and number of subsequences appearing only once
      sum_transitions[s_p] = 0
      for s_i in transitions[s_p] : 
        if transitions[s_p][s_i] == 1 : number_subseq_one_occ += 1
        sum_transitions[s_p] += transitions[s_p][s_i]
      
      #Unconditional probabilities
      unconditional_probabilities[s_p] = sum_transitions[s_p]/number_of_subseq
      
      # Conditional probability
      conditional_probabilities[s_p] = {}
      for s_i in transitions[s_p] : conditional_probabilities[s_p][s_i] = transitions[s_p][s_i]/sum_transitions[s_p]


    H_p = 0

    for s_p in transitions : 
      for s_i in transitions[s_p] :

          H_p -= unconditional_probabilities[s_p]*conditional_probabilities[s_p][s_i]*np.log(conditional_probabilities[s_p][s_i])/np.log(number_of_flowers)

    H_p_corr = H_p + H_0*number_subseq_one_occ/number_of_subseq

  return(H_p, H_p_corr)


def vector_of_conditional_entropies(sequence, p=5) :


  H_0, H_0_corr = conditional_entropy_pth_order(sequence, 0)
  entropy_vector = [H_0]
  corrected_entropy_vector = [H_0_corr]

  for order in range (1,p+1) : 
    if len(sequence) > order : 
      H, H_corr = conditional_entropy_pth_order(sequence, order, H_0)
      entropy_vector.append(H)
      corrected_entropy_vector.append(H_corr)

  return(entropy_vector, corrected_entropy_vector)

def compute_R_and_O(corrected_entropy_vector, thresold = 0.05) :

  if len(corrected_entropy_vector) == 1 : 
    return(None, None)

  else : 

    p = 0
    delta_entropy = corrected_entropy_vector[p]-corrected_entropy_vector[p+1]

    while (p < len(corrected_entropy_vector)-2) and (delta_entropy>=thresold) :
      p = p+1
      delta_entropy = corrected_entropy_vector[p]-corrected_entropy_vector[p+1]

    return(1-corrected_entropy_vector[p], p)

def compute_entropy_dataframe(array_info_to_save, list_of_parameter_to_save,array_number,positions_DF,
  number_of_timesteps_in_time_window,number_of_bees,number_of_timesteps_per_sim,number_of_flowers) : 

  nrow = len(positions_DF.index)
  number_of_rows_to_process = number_of_bees*number_of_timesteps_in_time_window

  output_entropy = []
  sim = 0
  row = 0

  while (row + number_of_rows_to_process <= nrow) and (positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] == sim) : 

    visit_sequences = [[] for bee in range (number_of_bees)]
    last_positions = [None for bee in range (number_of_bees)]

    for i in range (row, row + number_of_rows_to_process) : 

      bee = positions_DF.loc[i,"bee"]
      position = positions_DF.loc[i,"position"]

      if (not np.isnan(position)) and (position != last_positions[bee]) : 

        visit_sequences[bee].append(int(position))
        last_positions[bee] = position

    row_of_dataframe = list(positions_DF.iloc[row,:3])

    for bee in range (number_of_bees) : 

      entropy_vector, corrected_entropy_vector = vector_of_conditional_entropies(visit_sequences[bee])
      R, O = compute_R_and_O(corrected_entropy_vector)
      output_entropy.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[bee, R, O])

    row += number_of_rows_to_process

    if row+number_of_rows_to_process <= nrow and positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] != sim : 

      sim = sim+1
      row = sim*number_of_timesteps_per_sim*number_of_bees

  return(output_entropy)




################################# SIMILARITY #################################

def similarity(seq1,seq2,size=3):

  # Give the similarity index (between 0 and 1) between seq1 and seq2 by comparing the number of subsequences of size 'size' in common
  # A sequence is a list of visitations to flowers, starting with the nest and ending with the nest

  if (seq1 is None or seq2 is None)  : # Special case where one of the sequences is None
    return(None)

  else :

    # Exclude visits to the nest
    seq1 = seq1[1:-1] 
    seq2 = seq2[1:-1]

    l1 = len(seq1)
    l2 = len(seq2)

    if (l1<size or l2<size): # If one of the sequences is smaller than 'size', similarity = 0
        return(0)

    else : 

        distinct_subseq = [{} for i in range (2)] # Dictionary where each entry is a different subsequence present in seq1 or seq2    
        subseq_in_common = {} # Dictionary of subsequences present in both sequences 

        number_of_subseq1 = l1 - size +1 # total number of subsequences (not unique) in seq1
        number_of_subseq2 = l2 - size +1

        sequences = [seq1,seq2]
        number_of_subseq = [number_of_subseq1,number_of_subseq2]

        # read both sequences and fnd the unique subsequences
        for i in range (2) : 
            for k in range (number_of_subseq[i]) :

                subseq = tuple(sequences[i][k:(k+size)]) # subsequence k of sequence i
                distinct_subseq[i][subseq] = subseq in distinct_subseq[1-i] # Note if this subsequence was discovered in the other seq
                if distinct_subseq[i][subseq] : subseq_in_common[subseq] = True # if it's the case, put in the subseq_in_common dict



        visitations_in_common = [[False for k in range (l1)],[False for k in range (l2)]] # Put True every time a flower was in the subsequences in common 

        for i in range (2) : 
            for k in range (number_of_subseq[i]) : 

                subseq = tuple(sequences[i][k:(k+size)])
                if subseq in subseq_in_common : visitations_in_common[i][k:(k+size)]=[True for k in range (size)]

        sab=np.sum(visitations_in_common[0])+np.sum(visitations_in_common[1])
        return(round(sab/(2*max(l1,l2)),8))
    

def compute_route_similarity_of_sim(list_of_visit_sequences, index=1) :

  number_of_bouts = len(list_of_visit_sequences)

  if number_of_bouts <= index : 

    return(None)

  else : 

    number_of_bees = len(list_of_visit_sequences[0]) 
    list_of_route_similarities = []

    for bout in range (number_of_bouts-index) :
      
      current_row = list_of_visit_sequences[bout]
      next_row = list_of_visit_sequences[bout+index]

      
      similarity_for_bee = []

      for bee in range(number_of_bees) :

        current_sequence_bee = current_row[bee]
        next_sequence_bee = next_row[bee]
        similarity_for_bee.append(similarity(current_sequence_bee,next_sequence_bee))

      list_of_route_similarities.append(similarity_for_bee)

    return(list_of_route_similarities)


def compute_sim_afterwards(array_info_to_save, list_of_parameter_to_save,array_number,matrix_of_visit_seq, index) : 

  nrow = len(matrix_of_visit_seq.index)
  output = []

  nsim = np.max(matrix_of_visit_seq[0])
  nbee = np.max(matrix_of_visit_seq[2])

  row = 0
  
  for sim in range (nsim+1) : 
    for bee in range (nbee+1) : 


      data = matrix_of_visit_seq[(matrix_of_visit_seq[0] == sim)&(matrix_of_visit_seq[2] == bee)].sort_values(1)
      nbouts = len(data)

      for bout in range(nbouts - index) : 


        visit_seq_1 = np.array(data.iloc[bout,3:])
        visit_seq_1 = visit_seq_1[visit_seq_1>=0] # remove all the -1  

        visit_seq_2 = np.array(data.iloc[bout+index,3:])
        visit_seq_2 = visit_seq_2[visit_seq_2>=0] # remove all the -1     

        simi = similarity(visit_seq_1,visit_seq_2)

        output.append(array_info_to_save+list_of_parameter_to_save+[array_number, sim, bout, bee, simi])

  return(output)

def compute_sim_patch(array_info_to_save, list_of_parameter_to_save,array_number,matrix_of_visit_seq, patch_dict) :

  nrow = len(matrix_of_visit_seq.index)
  output = []

  nsim = np.max(matrix_of_visit_seq[0])
  nbee = np.max(matrix_of_visit_seq[2])

  row = 0

  
  for sim in range (nsim+1) : 
    for bee in range (nbee+1) : 


      data = matrix_of_visit_seq[(matrix_of_visit_seq[0] == sim)&(matrix_of_visit_seq[2] == bee)].sort_values(1).reset_index()
      nbouts = len(data)

      for bout in range(nbouts - 1) : 


        visit_seq_1 = np.array(data.iloc[bout,3:])
        visit_seq_1 = visit_seq_1[visit_seq_1>=0] # remove all the -1  

        visit_seq_2 = np.array(data.iloc[bout+1,3:])
        visit_seq_2 = visit_seq_2[visit_seq_2>=0] # remove all the -1  


        # Transform the visit seq to get the patches visited
        visit_seq_1 = np.array([patch_dict[flower] for flower in visit_seq_1])
        visit_seq_2 = np.array([patch_dict[flower] for flower in visit_seq_2])
        diff1 = np.concatenate(([True], np.diff(visit_seq_1) != 0))
        diff2 = np.concatenate(([True], np.diff(visit_seq_2) != 0))
        visit_seq_1, visit_seq_2 = visit_seq_1[diff1], visit_seq_2[diff2]

        sim_patch = similarity(visit_seq_1,visit_seq_2)


        output.append(array_info_to_save+list_of_parameter_to_save+[array_number, sim, bout, bee, sim_patch])

  return(output)

################################# LOCAL COMPETITION #################################

def local_competition(flowers_visited,number_of_bees) :

  flowers_visited = flowers_visited[:,1:] # Exclude the nest
  bee_indices = np.array([bee for bee in range (number_of_bees)])
  local_competitions=[]

  if number_of_bees == 1 : 
    return([0.])

  for bee in range (number_of_bees) :

    did_bee_visit_this_flower = flowers_visited[bee,:]>0
    number_of_distinct_patches_visited = np.sum(did_bee_visit_this_flower)

    if number_of_distinct_patches_visited == 0 :

      local_competitions.append(0)

    else : 

      number_of_visits_of_bee = np.sum(flowers_visited[bee,:]>0)
      number_of_visits_other_bees = np.sum(flowers_visited[:,did_bee_visit_this_flower][bee_indices!=bee,:])
      local_competitions.append(np.round(number_of_visits_other_bees/(number_of_visits_of_bee*number_of_distinct_patches_visited),8))

  return(local_competitions)

def compute_local_competition_dataframe(array_info_to_save, list_of_parameter_to_save,array_number,positions_DF,
  number_of_timesteps_in_time_window,number_of_bees,number_of_timesteps_per_sim,number_of_flowers) : 

  nrow = len(positions_DF.index)
  number_of_rows_to_process = number_of_bees*number_of_timesteps_in_time_window
  output_local_competition = []
  sim = 0
  row = 0

  while (row + number_of_rows_to_process <= nrow) and (positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] == sim) : 

    flowers_visited = np.zeros((number_of_bees,number_of_flowers))
    last_positions = [None for bee in range (number_of_bees)]

    for i in range (row, row + number_of_rows_to_process) : 

      bee = positions_DF.loc[i,"bee"]
      position = positions_DF.loc[i,"position"]

      if (not np.isnan(position)) and (position != last_positions[bee]) : 

        flowers_visited[bee,int(position)]+=1
        last_positions[bee] = position

    local_competitions = local_competition(flowers_visited,number_of_bees)

    row_of_dataframe = list(positions_DF.iloc[row,:3])

    for bee in range (number_of_bees) : 

      local_competition_bee = local_competitions[bee]
      output_local_competition.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[bee,local_competition_bee])

    row += number_of_rows_to_process

    if row+number_of_rows_to_process <= nrow and positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] != sim : 

      sim = sim+1
      row = sim*number_of_timesteps_per_sim*number_of_bees

  return(output_local_competition)




################################# TIME BETWEEN VISITS #################################

def compute_time_between_visits(array_info_to_save,list_of_parameter_to_save,array_number,positions_DF,
  number_of_timesteps_in_time_window,number_of_bees,number_of_timesteps_per_sim,number_of_flowers, different_bee = True) : 


  nrow = len(positions_DF.index)
  number_of_rows_to_process = number_of_bees*number_of_timesteps_in_time_window
  output_time_between_visits = []
  sim = 0
  row = 0

  while (row + number_of_rows_to_process <= nrow) and (positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] == sim) : 

    last_bee_on_flower = [None for flower in range (number_of_flowers)]
    last_time_visit = [0 for flower in range (number_of_flowers)]
    time_between_visits = []

    last_positions = [None for bee in range (number_of_bees)]

    for i in range (row, row + number_of_rows_to_process) : 

      bee = positions_DF.loc[i,"bee"]
      position = positions_DF.loc[i,"position"]
      time = positions_DF.loc[i,"time"]

      if (not np.isnan(position)) and (position != last_positions[bee]) : 

        position = int(position)

        if last_bee_on_flower[position] is not None : 
          if (different_bee and last_bee_on_flower[position]!=bee) or (not different_bee) : 
            time_between_visits.append(time - last_time_visit[position])


        last_positions[bee] = position
        last_bee_on_flower[position] = bee
        last_time_visit[position] = time

    if len(time_between_visits) == 0 :
      average_time_between_visits = None
    else : 
      average_time_between_visits = np.mean(time_between_visits)

    row_of_dataframe = list(positions_DF.iloc[row,:4])

    output_time_between_visits.append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe+[average_time_between_visits])

    row += number_of_rows_to_process

    if row+number_of_rows_to_process <= nrow and positions_DF.loc[row+number_of_rows_to_process-1,"simulation"] != sim : 

      sim = sim+1
      row = sim*number_of_timesteps_per_sim*number_of_bees

  return(output_time_between_visits)




################################# PLOTS AND OUTPUT DATAFRAMES #################################

def get_metrics_to_compute(metrics_to_compute, dataframes_already_computed, overwrite_files) : 

  if not overwrite_files : 
    copy_metrics_to_compute = [metric for metric in metrics_to_compute]
    for metric in copy_metrics_to_compute : 
      dataframe_name = metric + '.csv'
      if (dataframe_name in dataframes_already_computed) : metrics_to_compute.remove(metric)
  return() 

def update_output_list(array_path, metric, outputs, i, list_of_parameter_to_save, array_number, array_info_to_save):

  dataframe = pd.read_csv(array_path+'\\'+metric+'_DF.csv')
  nrow = len(dataframe.index)
  for row in range (nrow) : 
    row_of_dataframe = list(dataframe.iloc[row,:])
    outputs[i].append(array_info_to_save+list_of_parameter_to_save+[array_number]+row_of_dataframe)
  return()

def save_final_output(outputs, i, metric, parameters_to_save, name_of_array_info_to_save, dict_of_metrics, data_directory_path) :

  if metric == "entropy" : 
    last_col_names = ["timestep", "time", "bee", "R", "O"]

  elif metric == "nflowers" : 
    last_col_names = ["bout", "bee", "nflowers", "length_bout"]

  elif dict_of_metrics[metric] == "bout" : 
    last_col_names = ["bout","bee"] + [metric]
  
  else : 
    last_col_names = ["timestep", "time","bee"] + [metric]

  colnames = name_of_array_info_to_save + parameters_to_save+["array_number","sim"] + last_col_names
  dataframe = pd.DataFrame(outputs[i],columns = colnames).dropna()
  dataframe.to_csv(path_or_buf=data_directory_path+'\\'+metric+'.csv',index=False)
  return(dataframe)

def get_max_bout(data_directory_path,parameters_to_save,row_variable,col_variable,hue_variable, name_of_array_info_to_save, similarity) : 

  duration_sim = similarity.groupby(parameters_to_save+name_of_array_info_to_save+["array_number","sim"])["bout"].max().reset_index()
  
  if col_variable is None and row_variable is None and hue_variable is None : 
    max_bout = duration_sim["bout"].min() + 1

  else : 

    if col_variable is None and row_variable is None and hue_variable is not None : 
      max_bout = duration_sim.groupby([hue_variable])["bout"].min().reset_index()

    elif col_variable is None and row_variable is not None and hue_variable is None : 
      max_bout = duration_sim.groupby([row_variable])["bout"].min().reset_index()

    elif col_variable is None and row_variable is not None and hue_variable is not None : 
      max_bout = duration_sim.groupby([row_variable,hue_variable])["bout"].min().reset_index() 

    elif col_variable is not None and row_variable is None and hue_variable is None :
      max_bout = duration_sim.groupby([col_variable])["bout"].min().reset_index()

    elif col_variable is not None and row_variable is None and hue_variable is not None :
      max_bout = duration_sim.groupby([hue_variable,col_variable])["bout"].min().reset_index()

    elif col_variable is not None and row_variable is not None and hue_variable is None : 
      max_bout = duration_sim.groupby([row_variable,col_variable])["bout"].min().reset_index()

    elif col_variable is not None and row_variable is not None and hue_variable is not None : 
      test = duration_sim.groupby([row_variable,col_variable,hue_variable, "array_number"])["bout"].min().reset_index()
      max_bout = duration_sim.groupby([row_variable,col_variable,hue_variable])["bout"].min().reset_index()

    max_bout["bout"] += 1
   
  return(max_bout)

def pretreating(metric, dataframe, max_bout, row_variable, row_variable_values, col_variable, col_variable_values,
 hue_variable, hue_variable_values) : 
  
  print("Pre treating data")

  # Had to adda "kee row" columns to decide which rows to keep, because I iterate over the different row/col/hue values of the plots
  # And I needed to keep track of the "good" and "bad" rows
  dataframe["keep_row"] = False 

  if col_variable is None and row_variable is None and hue_variable is None : 
    
    nbouts = max_bout
    if metric == "similarity_1" : nbouts = nbouts - 1
    if metric == "similarity_2" : nbouts = nbouts - 2
    if metric == "similarity_3" : nbouts = nbouts - 3
    dataframe["keep_row"] = dataframe["bout"] <= nbouts

  elif col_variable is None and row_variable is None and hue_variable is not None : 

    for hue_value in hue_variable_values : 
      nbouts = max_bout[max_bout[hue_variable]==hue_value].iloc[0,-1]
      if metric == "similarity_1" : nbouts = nbouts - 1
      if metric == "similarity_2" : nbouts = nbouts - 2
      if metric == "similarity_3" : nbouts = nbouts - 3
      dataframe.loc[(dataframe[hue_variable]==hue_value) & (dataframe["bout"]<=nbouts),"keep_row"] = True

  elif col_variable is None and row_variable is not None and hue_variable is None : 

    for row_value in row_variable_values :
      nbouts = max_bout[max_bout[row_variable]==row_value].iloc[0,-1]
      if metric == "similarity_1" : nbouts = nbouts - 1
      if metric == "similarity_2" : nbouts = nbouts - 2
      if metric == "similarity_3" : nbouts = nbouts - 3
      dataframe.loc[(dataframe[row_variable]==row_value) & (dataframe["bout"]<=nbouts), "keep_row"] = True

  elif col_variable is None and row_variable is not None and hue_variable is not None : 

    for row_value in row_variable_values :
      for hue_value in hue_variable_values : 
        nbouts = max_bout[(max_bout[row_variable]==row_value) & (max_bout[hue_variable]==hue_value) ].iloc[0,-1] 
        if metric == "similarity_1" : nbouts = nbouts - 1
        if metric == "similarity_2" : nbouts = nbouts - 2
        if metric == "similarity_3" : nbouts = nbouts - 3
        dataframe.loc[(dataframe[row_variable]==row_value) & (dataframe[hue_variable]==hue_value) & (dataframe["bout"]<=nbouts),"keep_row"] = True

  elif col_variable is not None and row_variable is None and hue_variable is None :

    for col_value in col_variable_values :
      nbouts = max_bout[max_bout[col_variable]==col_value].iloc[0,-1]
      if metric == "similarity_1" : nbouts = nbouts - 1
      if metric == "similarity_2" : nbouts = nbouts - 2
      if metric == "similarity_3" : nbouts = nbouts - 3
      dataframe.loc[(dataframe[col_variable]==col_value) & (dataframe["bout"]<=nbouts), "keep_row"] = True

  elif col_variable is not None and row_variable is None and hue_variable is not None :

    for col_value in col_variable_values :
      for hue_value in hue_variable_values : 
        nbouts = max_bout[(max_bout[col_variable]==col_value) & (max_bout[hue_variable]==hue_value) ].iloc[0,-1] 
        if metric == "similarity_1" : nbouts = nbouts - 1
        if metric == "similarity_2" : nbouts = nbouts - 2
        if metric == "similarity_3" : nbouts = nbouts - 3
        dataframe.loc[(dataframe[col_variable]==col_value) & (dataframe[hue_variable]==hue_value) & (dataframe["bout"]<=nbouts), "keep_row"] = True

  elif col_variable is not None and row_variable is not None and hue_variable is None : 

    for col_value in col_variable_values :
      for row_value in row_variable_values : 
        nbouts = max_bout[(max_bout[col_variable]==col_value) & (max_bout[row_variable]==row_value) ].iloc[0,-1] 
        if metric == "similarity_1" : nbouts = nbouts - 1
        if metric == "similarity_2" : nbouts = nbouts - 2
        if metric == "similarity_3" : nbouts = nbouts - 3
        dataframe.loc[(dataframe[col_variable]==col_value) & (dataframe[row_variable]==row_value) & (dataframe["bout"]<=nbouts), "keep_row"] = True

  elif col_variable is not None and row_variable is not None and hue_variable is not None : 

    for col_value in col_variable_values :
      for row_value in row_variable_values : 
        for hue_value in hue_variable_values : 
          nbouts = max_bout[(max_bout[col_variable]==col_value) & (max_bout[row_variable]==row_value) & (max_bout[hue_variable]==hue_value)].iloc[0,-1] 
          if metric == "similarity_1" : nbouts = nbouts - 1
          if metric == "similarity_2" : nbouts = nbouts - 2
          if metric == "similarity_3" : nbouts = nbouts - 3
          dataframe.loc[(dataframe[col_variable]==col_value) & (dataframe[row_variable]==row_value) & (dataframe[hue_variable]==hue_value) & (dataframe["bout"]<=nbouts), "keep_row"] = True

  dataframe = dataframe[dataframe["keep_row"]]

  return(dataframe)