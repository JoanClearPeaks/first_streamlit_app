import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import scipy
import copy
import altair as alt
import pytz
import plotly.graph_objects as go
import numpy as np
# from croniter import croniter    

#------------------------------ TITLE & DESCRIPTION ------------------------------------------------------------------------------------------------------------------------------------------------------
class Outlier_Quantiles():
  def outlier_quantiles(self): 
    st.title('DATAWASH_APP')
    st.divider()
    st.header("Outlier_Quantiles")
    st.markdown('''
                This module performs an outlier detection on a specific table and column. The detection is based on quantile calculations. 
                The module returns the total number of detected outliers, the total values that have been checked, and a scatter plot. 
                The scatter plot includes thresholds, clearly distinguishing between values that are considered outliers and those that are not.
                ''')
    
    
    # Lee el archivo CSV
    selection = pd.read_csv('predictive_maintenance.csv')
    selection.index = selection.index + 1
    
    first_row = selection.iloc[0,:]
    with st.expander(f"Show selection first row", expanded=False):
        st.dataframe(first_row, use_container_width=True)
        
    with st.expander(f"Show selecion first row", expanded=False):
        st.table(first_row)
        
    for col in selection.columns:
          if selection[col].dtype == 'object':
              try:
                  selection[col] = pd.to_datetime(selection[col])
              except ValueError:
                  pass

    # with st.expander('ORIGINAL DATA'):
    #   st.dataframe(selection)
    # selection['DATES'] = selection['DATES'].dt.date
    # st.write(selection.dtypes)
    #------------------------- USER COLUMN SELECTION -------------------------------------------------------------
    st.sidebar.subheader('COLUMN', help='''Please select the numerical target column.
                          The selection is limited to integer or float types.''')
    numeric_columns = selection.select_dtypes(include=['float','int']).columns.tolist()  
    
    def get_data():
        df = pd.DataFrame({
            "lat": np.random.randn(200) / 50 + 37.76,
            "lon": np.random.randn(200) / 50 + -122.4,
            "team": ['A','B']*100
        })
        return df
    
    if st.button('Generate new points'):
        st.session_state.df = get_data()
    if 'df' not in st.session_state:
        st.session_state.df = get_data()
    df = st.session_state.df
    
    with st.form("my_form"):
        header = st.columns([1,2,2])
        header[0].subheader('Color')
        header[1].subheader('Opacity')
        header[2].subheader('Size')
    
        row1 = st.columns([1,2,2])
        colorA = row1[0].color_picker('Team A', '#0000FF')
        opacityA = row1[1].slider('A opacity', 20, 100, 50, label_visibility='hidden')
        sizeA = row1[2].slider('A size', 50, 200, 100, step=10, label_visibility='hidden')
    
        row2 = st.columns([1,2,2])
        colorB = row2[0].color_picker('Team B', '#FF0000')
        opacityB = row2[1].slider('B opacity', 20, 100, 50, label_visibility='hidden')
        sizeB = row2[2].slider('B size', 50, 200, 100, step=10, label_visibility='hidden')
    
        st.form_submit_button('Update map')
    
    alphaA = int(opacityA*255/100)
    alphaB = int(opacityB*255/100)
    
    df['color'] = np.where(df.team=='A',colorA+f'{alphaA:02x}',colorB+f'{alphaB:02x}')
    df['size'] = np.where(df.team=='A',sizeA, sizeB)
    
    st.map(df, size='size', color='color')
      
    self.target_column = st.sidebar.selectbox("Select the target column", [None] +  list(numeric_columns), index=0, label_visibility="collapsed")
    
    if self.target_column == None:
        st.sidebar.info('Please select a valid target column')
        return
    st.sidebar.subheader('DATE COLUMN', help= '''Please select the date column.
                          The selection is limited to date type.''')        
    date_columns = selection.select_dtypes(include=['datetime']).columns.tolist()
    self.date_column = st.sidebar.selectbox("Select the date column", [None] + ['False'] + list(date_columns), index=0, label_visibility="collapsed")


      
    # if self.date_column == None:
    #     st.sidebar.info('Please select a valid date column')
    #     return

        
        
    
    #------------------------------ PARAMETERS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader('PARAMETERS')
    
    self.warning_dic = {}
      
    if self.date_column != None:
      st.write(str(selection[self.date_column].dtype))
      if 'datetime' not in str(selection[self.date_column].dtype):
            st.sidebar.info('Please select a DATETIME column')
            st.write(selection[self.date_column].dtype)
            return 
      # st.write(selection[self.date_column].dtype.__name__)
      selection[self.date_column] = selection[self.date_column].replace('2019-09-11', '2020-01-11')
      selection[self.date_column] = selection[self.date_column].replace('2019-09-12', '2018-01-11')
      selection[self.date_column] = selection[self.date_column].replace('2019-09-24', '2019-09-23')
      selection.iloc[1] = ['2020-06-20', 303, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      selection.iloc[30] = ['2020-06-20', 303, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      selection.iloc[35] = ['2020-06-20', 298, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      selection.iloc[40] = ['2020-06-21', 270, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      selection.iloc[100] = ['2020-06-20', 390, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      selection.iloc[150] = ['2019-09-14', 390, 300, 1950, 50, 120, 0, 0, 0, 0, 0, 0]

      selection.iloc[37] = ['2020-06-21', 300, 300, 1200, 50, 120, 0, 0, 0, 0, 0, 0]
      

      selection[self.date_column] = selection[self.date_column].dt.date
      with st.expander('DATA CHANGE'):
        st.dataframe(selection[self.date_column])
        
      # selection[self.date_column] = pd.to_datetime(selection[self.date_column]).dt.date
      self.start_date1 = selection[self.date_column].min()
      self.end_date1 = selection[self.date_column].max()
      date_range = [date.date() for date in pd.date_range(start=self.start_date1, end=self.end_date1, freq='D') if date.date() in selection[self.date_column].tolist()]
      
      
      st.write("          ")
      st.sidebar.markdown("**Date Range**", help = '''Default Start Date: refers to the earliest date found in the selected date column.
                                                      \nDefault End Date: refers to the latest date found in the selected date column.''')
      # self.start_date = st.sidebar.selectbox("Select Start Date", [None] + date_range)
      
      self.start_date = st.sidebar.date_input(
      "Select Start Date", None,
      min_value=min(date_range),
      max_value=max(date_range)
      )
      # st.write('Initial start date', self.start_date)

      if self.start_date == None:
          st.sidebar.info('Please select a Start Date')
          return

      start_changed = False
      if self.start_date not in date_range:
        # Encontrar la fecha más cercana en el conjunto de datos
        self.start_date_prov = min(date for date in date_range if date > self.start_date)
        st.sidebar.info(f'Start Date {self.start_date} is not in your data. \n\nNearest Start Date selected: {self.start_date_prov}')
        self.start_date = self.start_date_prov
        start_changed = True
        
      self.end_date = st.sidebar.date_input(
      "Select End Date",
      None,
      min_value = min(date_range),
      max_value = max(date_range)
      )

      end_changed = False
      if self.end_date not in date_range:
        # Encontrar la fecha más cercana en el conjunto de datos
        self.end_date_prov = min(date for date in date_range if date > self.end_date)
        st.sidebar.info(f'End Date {self.end_date} is not in your data. \n\nNearest End Date selected: {self.end_date_prov}')
        self.end_date = self.end_date_prov
        end_changed = True

      if self.end_date == None:
          st.sidebar.info('Please select an End Date',)
          return 
      
      elif self.start_date == self.end_date:
          st.sidebar.warning('Start Date and End Date must be different')
          return 
      
      elif self.start_date > self.end_date:
          st.sidebar.warning('Start Date must be lower than End Date')
          return 

      

      

      # if self.start_date == self.end_date:
      #     st.sidebar.warning(f'Start Date (modified = {start_changed}) and End Date (modified = {end_changed}) must be different')
      #     return 
      
      # elif self.start_date > self.end_date:
      #     st.sidebar.warning(f'Start Date (modified = {start_changed}) must be lower than End Date (modified = {end_changed})')
      #     return  

      self.options = st.sidebar.radio(
                      "**Total Threshold or Selection Threshold**",
                      ["Total Threshold", "Selection Threshold"], horizontal=True, 
                      help = '''Total Threshold: Thresholds are calculated based on the entire dataset of values within the selected numerical column.\n
Selection Threshold: Thresholds are calculated based on the dataset of values within the selected numerical column that fall within the specified date range.''')

    if self.date_column != None:
      self.rolling_period =  st.sidebar.number_input('**Number of observations (mean)**', min_value=1, step=1, value = 1, help = '''
Select the grouping criterion for observations, where numerical values will be averaged based on the grouping.''') # max_value=self.diff_days
  
      if self.rolling_period == 0:
          st.sidebar.info('Please insert a rolling period')
          return
        
    self.sensitivity = st.sidebar.slider('**Sensitivity value**', 0.0, 3.0, step = 0.1, value=1.5, 
                                        help='''Multiplier that adjusts the sensitivity of outlier detection. Increasing the value reduces sensitivity, while decreasing it increases sensitivity.
                                        \nTip: It is recommended not to alter the default value of 1.50, as it is considered optimal for outlier detection.''') #info necessary

    if self.date_column != None:
        st.sidebar.markdown("**Enable grouping**", help='''If dates are repeated, their values will be averaged, resulting in each unique date having 
        a single numerical value representing the mean of all its previous values.''')
        enabled_group = st.sidebar.checkbox('Group per day')
        self.warning_dic["ENABLED_GROUPING_DAY"] = enabled_group

        
    st.sidebar.markdown("**Enable Warnings**")
    enabled = st.sidebar.checkbox('Visualize and persist warnings')
    self.warning_dic["ENABLED"] = enabled

    def check_outliers(df):
      if self.date_column != None:
          df['original_index'] = df.index
          df_filtered = df[(df[self.date_column] >= self.start_date) & (df[self.date_column] <= self.end_date)]
          df_filtered = df_filtered.sort_values(by=self.date_column).reset_index()
          st.header('Sort by date')
          st.dataframe(df_filtered)
          # df_filtered = df_filtered.reset_index(drop=True)
          # st.dataframe(df_filtered)
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
              with st.expander('Df filtered per day'):
                  st.dataframe(df_filtered)

          # Calcular la media para grupos de tres valores
          if self.rolling_period != 1:
              if self.rolling_period <= df_filtered.shape[0]:
                  df_filtered.rename(columns={'index': 'original_index'}, inplace=True)
                  df_filtered.reset_index(drop=True, inplace=True)
                
                  df_filtered[str(self.target_column)+'_MEAN'] = df_filtered[self.target_column].rolling(window=self.rolling_period, min_periods=1).mean()
                  # df_filtered[str(self.target_column)+'_MEAN'] = df_filtered[self.target_column].rolling(window=self.rolling_period, min_periods=1).mean()
              else:
                  st.warning('Rolling Period must be equal to or lower than the difference between the End Date and the Start Date')
                  return 
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

          self.lower_threshold = round(self.lower_threshold,4)
          self.upper_threshold = round(self.upper_threshold,4)
          
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
    # self.df_outliers.index = self.df_outliers.index + 1      
    # self.df_result.index = self.df_result.index + 1 

    st.write("          ")
    st.subheader('Results')

    if self.date_column != None:
        self.total_rows_python = self.df_filtered_original.shape[0]
        if self.rolling_period > 1:
            col1, col2 = st.columns(2)
        else:
            col1, col2, col3 = st.columns(3)

        col1.metric("OUTLIERS DETECTED", self.outliers_count)
        
        
        if self.rolling_period > 1:
            if self.warning_dic["ENABLED_GROUPING_DAY"]:
                col1.metric(f"GROUPS OF {self.rolling_period} DAYS", self.df_result.shape[0], help = f"As it is grouped by day, each group will consist of {self.rolling_period} unique days without repetition. The mean of the values for repeated days will be calculated to simplify and represent each date with a single value." )
            else:
                col1.metric(f"GROUPS OF {self.rolling_period} OBSERVATIONS", self.df_result.shape[0], help = 'Each group will consist of three observations, with the possibility to have repeated dates.')
            
            col2.metric("DAYS CHECKED", self.df_filtered_original[self.date_column].nunique(), help = 'Unique days checked within the specified date range. If DAYS CHECKED equals TOTAL OBSERVATIONS, it indicates that there are no repeated days.')
            col2.metric(f"TOTAL OBSERVATIONS", self.total_rows_python, help = 'Total observations/rows checked within the specified date range') #self.df.shape[0] to have only total days that have been grouped
        
        else:
            col2.metric("DAYS CHECKED", self.df_filtered_original[self.date_column].nunique(), help = 'Unique days checked within the specified date range. If DAYS CHECKED equals TOTAL OBSERVATIONS, it indicates that there are no repeated days.')
            col3.metric(f"TOTAL OBSERVATIONS", self.total_rows_python, help = 'Total observations/rows checked within the specified date range') #self.df.shape[0] to have only total days that have been grouped

    else:
        col1, col2 = st.columns(2)
        col1.metric("OUTLIERS DETECTED", self.outliers_count)
        col2.metric(f"TOTAL OBSERVATIONS", self.df_result.shape[0], help = 'Total observations/rows checked in your dataset')














      
    
     
  #---------------------------------------WARNINGS--------------------------------
    now = datetime.now(pytz.timezone('Europe/Madrid')).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    if self.warning_dic['ENABLED']:
      if self.outliers_count != 0:
          if self.outliers_count == 1:
              st.warning(f"There is {str(self.outliers_count)} outlier in {now} execution.", icon = '⚠')
          else:
              st.warning(f"There are {str(self.outliers_count)} outliers in {now} execution.", icon = '⚠')
          if self.date_column != None:
              if self.rolling_period == 1:
                  if not self.warning_dic["ENABLED_GROUPING_DAY"]:
                      copy_selection = copy.copy(selection)
                      copy_selection['original_index'] = copy_selection.index
        
                      self.df_outliers.rename(columns={f'{self.target_column}_VALUE': self.target_column}, inplace=True)
                      merged_df = pd.merge(copy_selection, self.df_outliers, on=[f'{self.target_column}', self.date_column])
                      df_uniques = merged_df.drop_duplicates(subset='original_index', keep='first')
                      indices = df_uniques['original_index'].tolist()
                      copy_selection.drop('original_index', axis=1, inplace=True)
                      self.df_outliers = copy_selection.loc[indices]
                      for index, row in self.df_outliers.iterrows(): 
                          st.write(f"Observation {index}, dated {row[self.date_column]}, with a value of {row[self.target_column]} in the target column {self.target_column}. The sensitivity was {self.sensitivity} and the threshold range ({self.lower_threshold} - {self.upper_threshold}) has been crossed.")
                  else:
                      for index, row in self.df_outliers.iterrows(): 
                          count_values = (selection[self.date_column] == row[self.date_column]).sum()
                          st.write(f"Group of {count_values} observations dated {row[self.date_column]}, with a value of {row[str(self.target_column)+'_VALUE']} in the target column {self.target_column}. The sensitivity was {self.sensitivity} and the threshold range ({self.lower_threshold} - {self.upper_threshold}) has been crossed.")
                      
              elif not self.warning_dic["ENABLED_GROUPING_DAY"]:
                  for index, row in self.df_outliers.iterrows(): 
                      st.write(f"Group with {self.rolling_period} observations between data range ({self.start_date} - {self.end_date}), with a mean value of {row[str(self.target_column)+'_MEAN']} in the target column {self.target_column}. The sensitivity was {self.sensitivity} and the threshold range ({self.lower_threshold} - {self.upper_threshold}) has been crossed.")
              else:
                  for index, row in self.df_outliers.iterrows(): 
                      st.write(f"Group with {self.rolling_period} days between data range ({self.start_date} - {self.end_date}), with a mean value of {row[str(self.target_column)+'_MEAN']} in the target column {self.target_column}. The sensitivity was {self.sensitivity} and the threshold range ({self.lower_threshold} - {self.upper_threshold}) has been crossed.")
          else:
               for index, row in self.df_outliers.iterrows(): 
                    st.write(f"Observation {index}, with a value of {row[self.target_column]} in the target column {self.target_column}. The sensitivity was {self.sensitivity} and the threshold range ({self.lower_threshold} - {self.upper_threshold}) has been crossed.")
        
            # Muestra las filas seleccionadas
            # st.dataframe(filas_seleccionadas)
          
          
          with st.expander(f"See outliers in {self.target_column} column", expanded=False):
              st.dataframe(self.df_outliers, use_container_width=True)
          
      else:
          st.success("No outliers have been detected.", icon = '✔')















      
  #---------------------------------------VISUALIZE--------------------------------
    if self.date_column != None:
        a = copy.copy(self.df_result[self.date_column])
        if self.rolling_period == 1:
            a = pd.to_datetime(a, format="%Y-%m-%dT%H:%M:%S.%fZ")
            self.df_result[self.date_column] = a.apply(lambda x: x.strftime("%Y-%m-%d"))
        
        else:
            self.df_result[self.date_column] = self.df_result[self.date_column].astype('category')
        
        # self.df_result[self.date_column] = self.df_result[self.date_column].apply(lambda x: datetime.strptime(x, "%y-%m-%d").strftime("%d/%m/%y"))
    container = st.container()
    if self.date_column == None:
      self.df_result = self.df_result.reset_index()
      self.df_result.rename(columns={'index': 'observation'}, inplace=True)

    x_column = list(self.df_result.columns)[0]
    y_column = list(self.df_result.columns)[1]
    outlier_column = 'OUTLIER'
    
    min_y = self.df_result[y_column].min()
    max_y = self.df_result[y_column].max()
    
    if max_y < self.upper_threshold:
        max_y = self.upper_threshold
    
    if min_y > self.lower_threshold:
        min_y = self.lower_threshold
    
    y_range=max_y-min_y
    padding_percentage = 0.05  # Puedes ajustar este valor según tus preferencias
    padding = y_range * padding_percentage          
        # Asumiendo que tienes las variables lower_threshold y upper_threshold definidas
    
    if self.date_column != None:
    # Crear el gráfico de Altair con líneas de umbrales y sombreado
        chart = alt.Chart(self.df_result).mark_point().encode(
            x=alt.X(f'{x_column}:N', axis=alt.Axis(labelFontSize=10)),
            y=alt.Y(f'{y_column}:Q', scale=alt.Scale(domain=[min_y - padding, max_y + padding]), axis=alt.Axis(title=y_column)),  # Ajustar el dominio del eje y
            color=alt.condition(
                alt.datum[outlier_column],
                alt.value('red'),  # Si OUTLIER es True, color rojo
                alt.value('green')  # Si OUTLIER es False, color verde
            ),
            tooltip=[
                f'{x_column}:N',
                f'{y_column}:Q',
                f'{outlier_column}:N'
            ]
        ).properties(width=600, height=400)
    
    else:
        chart = alt.Chart(self.df_result).mark_point().encode(
            x=f'observation:N',
            y=alt.Y(f'{y_column}:Q', scale=alt.Scale(domain=[min_y - padding, max_y + padding]), axis=alt.Axis(title=y_column)),  # Ajustar el dominio del eje y
            color=alt.condition(
                alt.datum[outlier_column],
                alt.value('red'),  # Si OUTLIER es True, color rojo
                alt.value('green')  # Si OUTLIER es False, color verde
            ),
            tooltip=[
                'observation:N',
                f'{y_column}:Q',
                f'{outlier_column}:N'
            ]
        ).properties(width=600, height=400)
    
    # Líneas de umbrales con leyenda y ajuste de dominio del eje y
    
    # lower_threshold_line = alt.Chart(pd.DataFrame({'Lower Threshold': [self.lower_threshold]})).mark_rule(color='blue', strokeWidth=1.5).encode(
    # y=alt.Y('Lower Threshold:Q'),
    # tooltip=alt.Tooltip('Lower Threshold:Q')
    # )

    # upper_threshold_line = alt.Chart(pd.DataFrame({'Upper Threshold': [self.upper_threshold]})).mark_rule(color='blue', strokeWidth=1.5).encode(
    #     y=alt.Y('Upper Threshold:Q'),
    #     tooltip=alt.Tooltip('Upper Threshold:Q')
    # )

    lower_threshold_line = alt.Chart(pd.DataFrame({'Lower Threshold': [self.lower_threshold]})).mark_rule(color='blue', strokeWidth=1.5).encode(
    y=alt.Y('Lower Threshold:Q'),
    tooltip=alt.Tooltip(['Lower Threshold:Q'], format='.2f', title='Lower Threshold')
    )
    
    upper_threshold_line = alt.Chart(pd.DataFrame({'Upper Threshold': [self.upper_threshold]})).mark_rule(color='blue', strokeWidth=1.5).encode(
        y=alt.Y('Upper Threshold:Q'),
        tooltip=alt.Tooltip(['Upper Threshold:Q'], format='.2f', title='Upper Threshold')
    )


    
    
    if self.date_column == None: 
    # Sombreado entre umbrales
        shaded_area = alt.Chart(self.df_result).mark_area(opacity=0.15, color='yellow').encode(
            x='observation:N',
            y='lower_threshold:Q',
            y2='upper_threshold:Q'
            # tooltip=[
            #     f'{x_column}:N',
            #     f'{y_column}:Q',
            #     f'{outlier_column}:N'
            # ]
        ).transform_calculate(
            lower_threshold=f"{self.lower_threshold}",
            upper_threshold=f"{self.upper_threshold}"
        )
    
    else:
        # Sombreado entre umbrales
        shaded_area = alt.Chart(self.df_result).mark_area(opacity=0.15, color='yellow').encode(
            x=f'{x_column}:N',
            y='lower_threshold:Q',
            y2='upper_threshold:Q'
            # tooltip=[
            #     f'{x_column}:N',
            #     f'{y_column}:Q',
            #     f'{outlier_column}:N'
            # ]
        ).transform_calculate(
            lower_threshold=f"{self.lower_threshold}",
            upper_threshold=f"{self.upper_threshold}"
        )
    
    # ----------------------- chart2 -----------------------------
    
    container2 = st.container()
    final_chart2 = alt.layer(chart, lower_threshold_line, upper_threshold_line, shaded_area).interactive()
    
    if self.date_column == None:
        final_chart2 = final_chart2.configure_view(
            stroke=None
        ).properties(
            title=f'Scatter Plot of outliers in {self.target_column} column'
        )
    else:
         final_chart2 = final_chart2.configure_view(
            stroke=None
        ).properties(
            title=f'Scatter Plot of outliers between {self.start_date}/{self.end_date}'
        )
    
    with container2:
        st.altair_chart(final_chart2, use_container_width=True)

    #-------------------------------------PLOTLY PLOT-------------------------------------------
    
    
    # Supongamos que self.df_result es tu DataFrame y que x_column, y_column y outlier_column son tus columnas
    
    # Crear el scatter plot
    # fig = go.Figure()
    
    # # Añadir los puntos al gráfico
    # fig.add_trace(go.Scatter(
    #     x=self.df_result[x_column] if self.date_column != 'False' else self.df_result.index,
    #     y=self.df_result[y_column],
    #     mode='markers',
    #     marker=dict(
    #         color=self.df_result[outlier_column].map({True: 'red', False: 'green'}),
    #         size=10
    #     ),
    #     text=self.df_result[outlier_column],
    #     name='Observations'
    # ))
    
    # # Añadir las líneas de umbrales
    # fig.add_shape(
    #     type='line',
    #     y0=self.lower_threshold,
    #     y1=self.lower_threshold,
    #     x0=self.df_result[x_column].min() if self.date_column != 'False' else self.df_result.index.min(),
    #     x1=self.df_result[x_column].max() if self.date_column != 'False' else self.df_result.index.max(),
    #     line=dict(color='blue', width=1.5),
    #     name='Lower Threshold'
    # )
    
    # fig.add_shape(
    #     type='line',
    #     y0=self.upper_threshold,
    #     y1=self.upper_threshold,
    #     x0=self.df_result[x_column].min() if self.date_column != 'False' else self.df_result.index.min(),
    #     x1=self.df_result[x_column].max() if self.date_column != 'False' else self.df_result.index.max(),
    #     line=dict(color='blue', width=1.5),
    #     name='Upper Threshold'
    # )
    
    # # Añadir el sombreado entre umbrales
    # fig.add_trace(go.Scatter(
    #     x=pd.concat([self.df_result[x_column] if self.date_column != 'False' else self.df_result.index, self.df_result[x_column] if self.date_column != 'False' else self.df_result.index[::-1]]),
    #     y=pd.concat([pd.Series([self.lower_threshold]*len(self.df_result)), pd.Series([self.upper_threshold]*len(self.df_result))[::-1]]),
    #     fill='toself',
    #     fillcolor='yellow',
    #     line=dict(color='yellow'),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    
    # # Configurar el layout del gráfico
    # fig.update_layout(
    #     title=f'Scatter Plot of outliers in {self.target_column} column' if self.date_column == 'False' else f'Scatter Plot of outliers between {self.start_date}/{self.end_date}',
    #     xaxis_title=x_column if self.date_column != 'False' else 'Observation',
    #     yaxis_title=y_column,
    #     autosize=False,
    #     width=600,
    #     height=400,
    #     margin=dict(l=50, r=50, b=100, t=100, pad=4)
    # )
    
    # # Mostrar el gráfico
    # fig.show()

    # st.write('HOLA')
    # Example DataFrames
    # df1 = pd.DataFrame({'A': [1, 3, 4,3], 'B': [4, 3, 6,3], 'F': [1, 2, 7,1]})
    # df2 = pd.DataFrame({'A': [7, 1, 4,3], 'B': [7, 4, 5,3], 'F': [5, 5, 5,5]})  # Assume this is your other DataFrame
    # df1['original_index'] = df1.index
    # st.dataframe(df1)
    # st.dataframe(df2)

    
    # merged_df = pd.merge(df1, df2, on=['A', 'B'])
    # st.dataframe(merged_df)
    # indices = merged_df['original_index'].tolist()
    # st.write(indices)
    # df1.drop('original_index', axis=1, inplace=True)
    # st.dataframe(df1)

    with st.expander('ORIGINAL'):
        st.dataframe(selection)
    with st.expander('RESULT'):
        st.dataframe(self.df_result)
    with st.expander('OUTLIERS'):
        st.dataframe(self.df_outliers, use_container_width=True)

    copia_original = copy.copy(selection)
    copia_original['original_index'] = copia_original.index
    if self.date_column != None:
        if self.warning_dic["ENABLED_GROUPING_DAY"]:
            if self.rolling_period == 1: 
                pass
                
        elif self.rolling_period == 1: 
            self.df_outliers.rename(columns={f'{self.target_column}_VALUE': self.target_column}, inplace=True)
            merged_df = pd.merge(copia_original, self.df_outliers, on=[f'{self.target_column}', self.date_column])
            st.dataframe(merged_df)
            df_uniques = merged_df.drop_duplicates(subset='original_index', keep='first')
        
            indices = df_uniques['original_index'].tolist()
            st.write(indices)
            copia_original.drop('original_index', axis=1, inplace=True)
        
            filas_seleccionadas = copia_original.loc[indices]
            
            # Muestra las filas seleccionadas
            st.dataframe(filas_seleccionadas)

        else:
            # Supongamos que 'df_original' es tu DataFrame original y 'df' es tu DataFrame agrupado
            dates = pd.to_datetime(selection[self.date_column])

            self.df_result[self.date_column] = self.df_result[self.date_column].str.split(' - ')
            
            # Inicializa una nueva columna para las listas de índices originales
            self.df_result['original_indices'] = None
            
            for i, row in self.df_result.iterrows():
                # st.header('Index: 1')
                # Convierte las cadenas de texto en objetos de fecha
                start_date = pd.to_datetime(row[self.date_column][0])
                end_date = pd.to_datetime(row[self.date_column][1])

                # st.write(selection[self.date_column].dt.date,start_date.date())
                # Encuentra las filas del DataFrame original que caen dentro del intervalo de fechas
                mask = (dates.dt.date >= start_date.date()) & (dates.dt.date <= end_date.date())
                # st.write(selection[self.date_column].dt.date >= start_date.date())
                # st.write(selection[self.date_column].dt.date <= end_date.date())
                # st.write('Mask:', mask)
                original_indices = selection[mask].index.tolist()
                # st.write(f'{start_date}-{end_date}')
                # st.write('Original Indeces:',original_indices)
            
                # Asigna la lista de índices originales a la nueva columna
                self.df_result.at[i, 'original_indices'] = original_indices

            st.dataframe(self.df_result)
            
            # Filtra las filas que son outliers
            outliers = self.df_result[self.df_result['OUTLIER'] == True]
            
            # Muestra los outliers y sus índices originales
            st.dataframe(outliers)
            # # Supongamos que 'df' es tu DataFrame
            # self.df_outliers[self.date_column] = self.df_outliers[self.date_column].str.split(' - ')
            
            # # Inicializa una nueva columna para las listas de fechas
            # self.df_outliers['date_list'] = None
            
            # for i, row in self.df_outliers.iterrows():
            #     # Convierte las cadenas de texto en objetos de fecha
            #     start_date = datetime.strptime(row[self.date_column][0], '%Y-%m-%d')
            #     end_date = datetime.strptime(row[self.date_column][1], '%Y-%m-%d')
            
            #     # Genera una lista de fechas para cada intervalo
            #     date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date-start_date).days + 1)]
            
            #     # Asigna la lista de fechas a la nueva columna
            #     self.df_outliers.at[i, 'date_list'] = date_list
            
            # # Muestra el DataFrame actualizado
            # st.dataframe(self.df_outliers)

            # # for date_intervals in self.df_outliers[self.date_column]:
            # #     date_range = [date.date() for date in pd.date_range(start=self.start_date1, end=self.end_date1, freq='D') if date.date() in selection[self.date_column].tolist()]
            
        
            
            
        
    
      
      


    return

  def matrix(self):
    import streamlit as st
    import pandas as pd
    import altair as alt
    
    # Datos
    data = {'A': [45, 37, 42, 35, 39],
            'B': [38, 31, 26, 28, 33],
            'C': [10, 15, 17, 21, 12]
            }
    
    df = pd.DataFrame(data)
    
    # Matriz de correlación
    corr_matrix = df.corr()
    
    # Crear gráfico de calor con Altair
    chart = alt.Chart(corr_matrix.reset_index().melt(id_vars='observation')).mark_rect().encode(
        x='observation:N',
        y='variable:N',
        color='value:Q'
    ).properties(
        width=400,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)


    import plotly.express as px
    fig = px.imshow(corr_matrix,
                labels=dict(x="Columnas", y="Columnas", color="Correlación"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r')

    fig.update_layout(title="Matriz de Correlación")
    
    st.plotly_chart(fig, use_container_width=True)

    return
  
a = Outlier_Quantiles()
a.outlier_quantiles()
# a.matrix()




