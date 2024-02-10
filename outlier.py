import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta


#------------------------------ TITLE & DESCRIPTION ------------------------------------------------------------------------------------------------------------------------------------------------------
class Outlier_Quantiles():
  def outlier_quantiles(): 
    st.divider()
    st.header("Outlier_Quantiles")
    st.markdown('''
                This module detects countries that are misspelled or do not really correspond to any country in the target column.
                It returns how many values are not a country, the number of total values checked, and a pie chart.
                ''')
    st.write("          ")
    
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

    return 

a = Outlier_Quantiles()
a.outlier_quantiles()



