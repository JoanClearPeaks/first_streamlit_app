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

    def check_outliers(df):
      if self.date_column != 'False':
          df_filtered = df[(df[self.date_column] >= self.start_date) & (df[self.date_column] <= self.end_date)]
          df_filtered = df_filtered.sort_values(by=self.date_column)
          df_filtered = df_filtered.reset_index(drop=True)
          
          self.df_filtered_original = copy.copy(df_filtered)
          
          if self.options == 'Total Threshold':
              # Calcular los cuantiles de la columna completa
              q1_full = df[self.target_column].quantile(0.25)
              q3_full = df[self.target_column].quantile(0.75)
          else:
              # Cuantiles por selección
              q1_full = df_filtered[self.target_column].quantile(0.25)
              q3_full = df_filtered[self.target_column].quantile(0.75)
          
                  
          if self.warning_dic["ENABLED_GROUPING_DAY"]:
              df_filtered = df_filtered.groupby(self.date_column)[self.target_column].mean().reset_index()
              # with st.expander('Df filtered per day'):
              #     st.dataframe(df_filtered)

          # Calcular la media para grupos de tres valores
          if self.rolling_period != 1:
              if self.rolling_period <= df_filtered.shape[0]:
                  df_filtered[str(self.target_column)+'_MEAN'] = df_filtered[self.target_column].rolling(window=self.rolling_period, min_periods=1).mean()
              else:
                  st.warning('Rolling Period must be equal to or lower than the difference between the End Date and the Start Date')
                  return False
          else:
              df_filtered[str(self.target_column)+'_VALUE'] = df_filtered[self.target_column].rolling(window=self.rolling_period, min_periods=1).mean()


          if self.rolling_period == 1:
              self.df_result = df_filtered.groupby(df_filtered.index // self.rolling_period).apply(lambda group: group.tail(1) if len(group) % self.rolling_period == 0 else pd.DataFrame())
                  
          # Crear grupos basados en el orden de la columna específica
          else:
              if df_filtered.shape[0]  % self.rolling_period != 0: #fa que et tregui aquella agrupació que no té el nombre d'elements agrupats passat per parametre (rolling period)
                  value = df_filtered.shape[0]  % self.rolling_period
                  df_filtered = df_filtered.drop(df_filtered.index[-value:])

              self.groups = df_filtered.groupby(df_filtered.index // self.rolling_period)
             
              first_dates = self.groups[self.date_column].first()
              last_dates = self.groups[self.date_column].last()
              # st.write(first_dates)
              # st.write(last_dates)

              # Aplicar la lógica para definir el rango de fechas
              # Obtener el DataFrame normal de first_dates y last_dates
              first_last_df = pd.DataFrame({'first_date': first_dates, 'last_date': last_dates})
              final_date_values = self.groups[str(self.target_column)+'_MEAN'].last().values                    
              first_last_df[str(self.target_column)+'_MEAN'] = final_date_values
              # Combina las fechas en un solo rango de fechas
              first_last_df[self.date_column] = first_last_df['first_date'].astype(str) + ' - ' + first_last_df['last_date'].astype(str)
              # st.dataframe(first_last_df)
              columns = [self.date_column,str(self.target_column)+'_MEAN']
              self.df_result = first_last_df[columns]
              
              

          self.df_result.reset_index(drop=True, inplace=True)

          # Calcular el rango intercuartílico (IQR)
          iqr = q3_full - q1_full

          # Definir umbrales para identificar outliers
          self.lower_threshold = q1_full - self.sensitivity * iqr
          self.upper_threshold = q3_full + self.sensitivity * iqr

          
          # Identificar outliers
          self.df_outliers = self.df_result[(self.df_result.iloc[:,-1] < self.lower_threshold) | (self.df_result.iloc[:,-1] > self.upper_threshold)]
          
          self.df_result = self.df_result[[self.date_column,list(self.df_result.columns)[-1]]]
          self.df_result['OUTLIER'] = np.where((self.df_result.iloc[:,-1] < self.lower_threshold) | (self.df_result.iloc[:,-1] > self.upper_threshold), True, False)
          self.df_outliers = self.df_outliers[[self.date_column,list(self.df_outliers.columns)[-1]]]
          self.df_outliers['LOWER_THRESHOLD'] = self.lower_threshold
          self.df_outliers['UPPER_THRESHOLD'] = self.upper_threshold

          
          self.df = copy.copy(df_filtered)

      else:
          q1_full = df[self.target_column].quantile(0.25)
          q3_full = df[self.target_column].quantile(0.75)
          iqr = q3_full - q1_full
          # Definir umbrales para identificar outliers
          self.lower_threshold = q1_full - self.sensitivity * iqr
          self.upper_threshold = q3_full + self.sensitivity * iqr
        
          # Identificar outliers
          self.df_outliers = df[(df[self.target_column] < self.lower_threshold) | (df[self.target_column] > self.upper_threshold)]
          df = df[[self.target_column]]
          df['OUTLIER'] = np.where((df[self.target_column] < self.lower_threshold) | (df[self.target_column] > self.upper_threshold), True, False)

          self.df_result = df

      return True

      if not check_outliers(selection):
          return

      self.outliers_count = self.df_outliers.shape[0]
      self.df_outliers.index = self.df_outliers.index + 1      
      self.df_result.index = self.df_result.index + 1 

      st.write("          ")
      st.subheader('Results')

      if self.date_column != 'False':
          self.total_rows_python = self.df_filtered_original.shape[0]
          if self.rolling_period > 1:
              col1, col2 = st.columns(2)
          else:
              col1, col2, col3 = st.columns(3)

          col1.metric("OUTLIERS DETECTED", self.outliers_count)
          
          
          if self.rolling_period > 1:
              if self.warning_dic["ENABLED_GROUPING_DAY"]:
                  col1.metric(f"GROUPS OF {self.rolling_period} DAYS", self.df_result.shape[0])
              else:
                  col1.metric(f"GROUPS OF {self.rolling_period} OBSERVATIONS", self.df_result.shape[0])
              
              col2.metric("DAYS CHECKED", self.df_filtered_original[self.date_column].nunique())
              col2.metric(f"TOTAL OBSERVATIONS", self.total_rows_python) #self.df.shape[0] to have only total days that have been grouped
          
          else:
              col2.metric("DAYS CHECKED", self.df_filtered_original[self.date_column].nunique())
              col3.metric(f"TOTAL OBSERVATIONS", self.total_rows_python) #self.df.shape[0] to have only total days that have been grouped

      else:
          col1, col2 = st.columns(2)
          col1.metric("OUTLIERS DETECTED", self.outliers_count)
          col2.metric(f"TOTAL OBSERVATIONS", self.df_result.shape[0])
    
    return 

a = Outlier_Quantiles()
a.outlier_quantiles()




