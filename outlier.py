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
        date_range = [date.date() for date in pd.date_range(start=self.start_date, end=self.end_date, freq='D') if date.date() in selection[self.date_column].tolist()]        
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
    
    return 

a = Outlier_Quantiles()
a.outlier_quantiles()




