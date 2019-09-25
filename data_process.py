import numpy as np 
import pandas as pd 
import daytime
import re 
from scipy.stats.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import timeit
import pickle 
import scipy 


####################################################################################################################################
############# CLEAN THE DATASET TO EXTRACT TRAJECTORIES ############################################################################
####################################################################################################################################

def data_processing():
  print("Loading the raw data file... ")
  # IMPORTING THE RAW DATA OF SUBOPTIMAL POLICY
  df = pd.read_csv('raw_data.txt', sep='\t', engine='python', nrows=1000, encoding= 'utf-16')
  print("/*************/")
  print("File loaded... ")

  # FUNCTIIONS USED TO CLEAN THE ABOVE RAW DATA 
  def read_data(file, enc='utf16', delim='\t', decim='.'):
      with open(file, newline='', encoding=enc) as datafile:
          return pd.read_csv(datafile, skiprows=0, header=0, delimiter=delim, decimal=decim, low_memory=False)


  def load_pkl(file):
      with open(file, 'rb') as inp:
          return pickle.load(inp)


  def save_pkl(var, file):
      with open(file, 'wb') as output:
          pickle.dump(var, output, pickle.HIGHEST_PROTOCOL)


  def running_mean(x, n):
      cumsum = np.cumsum(np.insert(x, 0, 0))
      return (cumsum[n:] - cumsum[:-n]) / float(n)

  def running_min(x, n):
      min_list = pd.DataFrame(x).rolling(n).min()
      return min_list.loc[(n-1):]

  def running_max(x, n):
      max_list = pd.DataFrame(x).rolling(n).max()
      return max_list.loc[(n-1):]

  def moving_avg_no_pad(pointList, winWidth):
      cumsum, moving_aves = [0], []
      #cumsum = [0] + list(accumulate(pointList))
      cumsum = np.cumsum(np.insert(pointList, 0, 0))
      for i, x in list(enumerate(pointList, 1)):
          if winWidth % 2 == 0:
              if i <= winWidth // 2:
                  moving_ave = (cumsum[i + winWidth // 2 - 1] - cumsum[0]) / (i + winWidth // 2 - 1)
                  moving_aves.append(moving_ave)
              elif i > len(pointList) - winWidth // 2 + 1:
                  moving_ave = (cumsum[len(pointList)] - cumsum[i - winWidth // 2 - 1]) / (
                              len(pointList) - i + winWidth // 2 + 1)
                  moving_aves.append(moving_ave)
              else:
                  moving_ave = (cumsum[i + winWidth // 2 - 1] - cumsum[i - winWidth // 2 - 1]) / winWidth
                  moving_aves.append(moving_ave)
          else:
              if i <= winWidth // 2:
                  moving_ave = (cumsum[i + winWidth // 2] - cumsum[0]) / (i + winWidth // 2)
                  moving_aves.append(moving_ave)
              elif i > len(pointList) - winWidth // 2:
                  moving_ave = (cumsum[len(pointList)] - cumsum[i - winWidth // 2 - 1]) / (
                              len(pointList) - i + winWidth // 2 + 1)
                  moving_aves.append(moving_ave)
              else:
                  moving_ave = (cumsum[i + winWidth // 2] - cumsum[i - winWidth // 2 - 1]) / winWidth
                  moving_aves.append(moving_ave)
      return moving_aves

  def range_sign(x, pos_thres=0):
      if x > pos_thres:
          return 1
      elif x < -pos_thres:
          return -1
      else:
          return 0


  def data_statistics_preprocess(df, target_column='Inc Electrical Antenna Tilt (deg)',drop_columns=[], pick_columns=[], split=0.7, reward_limit=20):
  	df = df.drop(columns=['CGI'])
  	#     df = df.dropna(how='any')
  	# Theoretical range of angle is considered as (0-25)
  	df['Electrical Antenna Tilt (deg)'] = df['Electrical Antenna Tilt (deg)'].apply(
  		lambda x: float(x) / 25)
  	# Theoretical range of throughput is considered as (4,300,000 Kbps)
  	df['Average UE PDCP DL Throughput (Kbps)'] = df['Average UE PDCP DL Throughput (Kbps)'].apply(
  		lambda x: float(x) / 4300000)
  	# Theoretical range of number of calls is considered as (3,200,000)
  	df['Num Calls'] = df['Num Calls'].apply(
  		lambda x: float(x) / 3200000)
  	# Theoretical range of number of calls is considered as (-1 to 499)
  	df['Time Advance Overshooting Factor'] = df['Time Advance Overshooting Factor'].apply(
  		lambda x: float(x) / 500)
  	# Theoretical range of number of calls is considered as (100)
  	df['Inter Site Distance (Km)'] = df['Inter Site Distance (Km)'].apply(
  		lambda x: float(x) / 100)

  	df['Var Electrical Antenna Tilt (deg)'] = df['Var Electrical Antenna Tilt (deg)']

  	# Theoretical range of number of calls is considered as (100)
  	df['Inc Electrical Antenna Tilt (deg)'] = df ['Inc Electrical Antenna Tilt (deg)'].apply(
  		lambda x: np.sign(x))
  	# Percentages normalised by 100
  	df['Low RSRP Samples Rate Edge (%)'] = df['Low RSRP Samples Rate Edge (%)'].apply(
  		lambda x: float(x) / 100)
  	df['Number of Times Interf (%)'] = df['Number of Times Interf (%)'].apply(
  		lambda x: float(x) / 100)
  	df['Number of Cells High Overlap High Rsrp Src Agg (%)'] = df['Number of Cells High Overlap High Rsrp Src Agg (%)']\
  		.apply(lambda x: float(x) / 100)
  	df['Number of Cells High Overlap High Rsrp Tgt Agg (%)'] = df['Number of Cells High Overlap High Rsrp Tgt Agg (%)']\
  		.apply(lambda x: float(x) / 100)
  	df['PDCCH CCE High Load (%)'] = df['PDCCH CCE High Load (%)'].apply(lambda x: float(x) / 100)

  	df.reset_index(drop=True, inplace=True)
  	return df


  def split_df(df, inputs=None, outputs=None):
      if outputs is None:
          raise Exception('Output cannot be None when splitting'.format())
      if inputs is None:
          return df.drop(outputs), df[outputs]
      return df[inputs], df[outputs]


  # OBTAIN THE CLEAN VERSION OF THE RAW DATA 
  clean_data = data_statistics_preprocess(df)


  print("columns:", clean_data.columns)


  cols = ["Electrical Antenna Tilt (deg)",
        "Average UE PDCP DL Throughput (Kbps)",
        "Num Calls",
        "Low RSRP Samples Rate Edge (%)",
        "FUZZY_LOW_RSRP_SAMPLES_EDGE_HIGH",
        "Number of Times Interf (%)",
        "FUZZY_NUM_TIMES_INTERF_HIGH",
        "Time Advance Overshooting Factor",
        "FUZZY_OSF_HIGH",
        "Number of Cells High Overlap High Rsrp Src Agg (%)",
        "FUZZY_NUM_CELLS_HIGH_OVERLAP_SRC_HIGH",
        "Number of Cells High Overlap High Rsrp Tgt Agg (%)",
        "FUZZY_NUM_TIMES_HIGH_OVERLAP_TGT_HIGH",
        "Inter Site Distance (Km)",
        "PDCCH CCE High Load (%)",
        "Inc Electrical Antenna Tilt (deg)",
        "Var Electrical Antenna Tilt (deg)",
         "Reward_Weighted_Delta_Driver_HighLevel_KPIs"]


  clean_data = clean_data[cols]
  clean_data = clean_data.dropna(how='any') 

  print("Shape of clean data:", clean_data.shape)


  # Create state space 
  def create_state(df):
      row_list = []
      row_list2 = []
      for index, rows in df.iterrows():
          my_list =[rows.iloc[1],
                   rows.iloc[2],
                   rows.iloc[3],
                   rows.iloc[4],
                   rows.iloc[5],
                   rows.iloc[6],
                   rows.iloc[7],
                   rows.iloc[8],
                   rows.iloc[9],
                   rows.iloc[10],
                   rows.iloc[11],
                   rows.iloc[12],
                   rows.iloc[13],
                   rows.iloc[14]]
          row_list.append(my_list) 
          row_list2.append(rows.iloc[17])
      df['state'] = row_list
      df['action'] = row_list2
      rl_df = df[["state", "action", "Reward_Weighted_Delta_Driver_HighLevel_KPIs"]]
      rl_df = rl_df.rename(columns={ "Reward_Weighted_Delta_Driver_HighLevel_KPIs":"reward"})
      return rl_df


  newdf = create_state(clean_data)
  newdf = newdf.reset_index()
  newdf = newdf.drop(columns=['index'])
  print("Shape of new df:", newdf.shape)

  return newdf


















