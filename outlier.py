import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta


#------------------------------ TITLE & DESCRIPTION ------------------------------------------------------------------------------------------------------------------------------------------------------
class Outlier_Quantiles():
  def outlier_quantiles(self): 
    st.title('DATAWASH_APP')
    st.divider()
    st.header("Outlier_Quantiles")
    st.markdown('''
                This module detects countries that are misspelled or do not really correspond to any country in the target column.
                It returns how many values are not a country, the number of total values checked, and a pie chart.
                ''')
    st.write("Hola ")
    
    # Lee el archivo CSV
    selection = pd.read_csv('predictive_maintenance.csv')
    
    #------------------------- USER COLUMN SELECTION -------------------------------------------------------------
    st.sidebar.subheader('COLUMN', help="Select the target column.")
    self.target_column = st.sidebar.selectbox("Select the target column", [None] +  list(selection.columns), index=0, label_visibility="collapsed")
    
    if self.target_column == None:
          st.sidebar.info('Please select a valid target column')
  #         return False
          
  #         elif 'int' not in str(selection[self.target_column].dtype) and 'float' not in str(selection[self.target_column].dtype):
  #             st.sidebar.warning('Please select a NUMERICAL column')
  #             st.write(selection[self.target_column].dtype)
  #             return False
  # st.sidebar.subheader('DATE COLUMN', help= "Select the date column")        
  #         self.date_column = st.sidebar.selectbox("Select the date column", [None] + ['False'] + list(selection.columns), index=0, label_visibility="collapsed")
          
  #         if self.date_column == None:
  #             st.sidebar.info('Please select a valid date column')
  #             return False
      
  #    #------------------------------ PARAMETERS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  #         st.sidebar.divider()
  #         st.sidebar.subheader('PARAMETERS')
    # col1, col2, col3 = st.columns(3)
    else:
      self.warning_dic = {}
      self.rolling_period = st.number_input('**Number of observations (mean)**', min_value=0, step=1)
      self.warning_dic["ENABLED_GROUPING_DAY"] = st.checkbox('Group per day')
      if self.rolling_period > 1:
          col1, col2 = st.columns(2)
      else:
          col1, col2, col3 = st.columns(3)
  
      col1.metric("OUTLIERS DETECTED", 1)
      
      
      if self.rolling_period > 1:
          if self.warning_dic["ENABLED_GROUPING_DAY"]:
              col1.metric(f"GROUPS OF 2 DAYS CHECKED", 10)
          else:
              col1.metric(f"GROUPS OF 2 OBSERVATIONS CHECKED", 10)
          
          col2.metric("DAYS CHECKED", 12)
          col2.metric(f"TOTAL OBSERVATIONS", 21) #self.df.shape[0] to have only total days that have been grouped
      
      else:
          col2.metric("DAYS CHECKED", 12)
          col3.metric(f"TOTAL OBSERVATIONS",21) #self.df.shape[0] to have only total days that have been grouped
      st.write('hola')
    return 

a = Outlier_Quantiles()
a.outlier_quantiles()



