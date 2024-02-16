import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import scipy


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
    
    
    # Lee el archivo CSV
    selection = pd.read_csv('predictive_maintenance.csv')
    st.dataframe(selection)
    #------------------------- USER COLUMN SELECTION -------------------------------------------------------------
    st.sidebar.subheader('COLUMN', help="Select the target column.")
    self.target_column = st.sidebar.selectbox("Select the target column", [None] +  list(selection.columns), index=0, label_visibility="collapsed")
    
    if self.target_column == None:
        st.sidebar.info('Please select a valid target column')
        return
    st.sidebar.subheader('DATE COLUMN', help= "Select the date column")        
    self.date_column = st.sidebar.selectbox("Select the date column", [None] + ['False'] + list(selection.columns), index=0, label_visibility="collapsed")
    
    if self.date_column == None:
        st.sidebar.info('Please select a valid date column')
        return
    #------------------------------ PARAMETERS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader('PARAMETERS')
    
    self.warning_dic = {}

    if self.date_column != 'False':
      self.start_date = selection[self.date_column].min()
      self.end_date = selection[self.date_column].max()
      date_range = [date.date() for date in pd.date_range(start=self.start_date, end=self.end_date, freq='D')]        
      st.write(self.start_date, self.end_date)
      st.write(date_range)
      st.write("          ")
      st.sidebar.markdown("**Date Range**")
      self.start_date = st.sidebar.selectbox("Select Start Date", [None] + date_range)
      if self.start_date == None:
          st.sidebar.info('Please select a Start Date')
          return 
      # filtered_date_range = [date for date in date_range if date >= self.start_date]
      self.end_date = st.sidebar.selectbox("Select End Date", [None] + date_range)
  
      if self.end_date == None:
          st.sidebar.info('Please select an End Date')
          return 
      elif self.start_date == self.end_date:
          st.sidebar.warning('Select different dates for both Start and End Date')
          return 
      
      elif self.start_date > self.end_date:
          st.sidebar.warning('Start Date must be lower than End Date')
          return 

      self.options = st.sidebar.radio(
                      "**Total Threshold or Selection Threshold**",
                      ["Total Threshold", "Selection Threshold"], horizontal=True)

    if self.date_column != 'False':
      self.rolling_period =  st.sidebar.number_input('**Number of observations (mean)**', min_value=0, step=1, value = 1) # max_value=self.diff_days
  
      if self.rolling_period == 0:
          st.sidebar.info('Please insert a rolling period')
          return False
        
    self.sensitivity = st.sidebar.slider('**Sensitivity value**', 0.0, 3.0, step = 0.1, value=1.5) #info necessary

    if self.date_column != 'False':
        st.sidebar.markdown("**Enable grouping**")
        enabled_group = st.sidebar.checkbox('Group per day')
        self.warning_dic["ENABLED_GROUPING_DAY"] = enabled_group

        
    st.sidebar.markdown("**Enable Warnings**")
    enabled = st.sidebar.checkbox('Visualize and persist warnings')
    self.warning_dic["ENABLED"] = enabled
    
    
    return 

a = Outlier_Quantiles()
a.outlier_quantiles()




