import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import scipy
import copy
import altair as alt
import pytz

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
    with st.expander('ORIGINAL DATA'):
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
    selection[self.date_column] = selection[self.date_column].replace('2019-09-11', '2020-01-11')
    with st.expander('DATA CHANGE'):
      st.dataframe(selection[self.date_column])
      
    if self.date_column != 'False':
      selection[self.date_column] = pd.to_datetime(selection[self.date_column]).dt.date
      self.start_date1 = selection[self.date_column].min()
      self.end_date1 = selection[self.date_column].max()
      date_range = [date.date() for date in pd.date_range(start=self.start_date1, end=self.end_date1, freq='D')]        
      
      st.write("          ")
      st.sidebar.markdown("**Date Range**")
      # self.start_date = st.sidebar.selectbox("Select Start Date", [None] + date_range)

      self.start_date = st.sidebar.date_input(
      "Select Start Date", None,
      min_value=min(date_range),
      max_value=max(date_range)
      )

      if self.start_date not in date_range and self.start_date != None:
        # Encontrar la fecha más cercana en el conjunto de datos
        self.start_date = min(date_range, key=lambda date: abs(date - self.start_date))

      st.write('Start Date:',self.start_date)
      
      if self.start_date == None:
          st.sidebar.info('Please select a Start Date')
          return 
      # filtered_date_range = [date for date in date_range if date >= self.start_date]
      # self.end_date = st.sidebar.selectbox("Select End Date", [None] + date_range)
      self.end_date = st.sidebar.date_input(
      "Select End Date",
      None,
      min_value = min(date_range),
      max_value = max(date_range)
      )

      if self.end_date not in date_range and self.end_date != None:
        # Encontrar la fecha más cercana en el conjunto de datos
        self.end_date = min(date_range, key=lambda date: abs(date - self.end_date))

      st.write('End Date:',self.end_date)
      
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
          return
        
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
          st.dataframe(df_filtered)
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
        
    
     
  #---------------------------------------WARNINGS--------------------------------
    now = datetime.now(pytz.timezone('Europe/Madrid')).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    if self.warning_dic['ENABLED']:
      if self.outliers_count != 0:
          if self.outliers_count == 1:
              st.warning(f"There is {str(self.outliers_count)} outlier in {now} execution.", icon = '⚠')
          else:
              st.warning(f"There are {str(self.outliers_count)} outliers in {now} execution.", icon = '⚠')

          with st.expander(f"See outliers in {self.target_column} column", expanded=False):
              st.dataframe(self.df_outliers, use_container_width=True)
      else:
          st.success("No outliers have been detected.", icon = '✔')
              
  #---------------------------------------VISUALIZE--------------------------------
    if self.date_column != 'False':
        a = copy.copy(self.df_result[self.date_column])
        if self.rolling_period == 1:
            a = pd.to_datetime(a, format="%Y-%m-%dT%H:%M:%S.%fZ")
            self.df_result[self.date_column] = a.apply(lambda x: x.strftime("%Y-%m-%d"))
        
        else:
            self.df_result[self.date_column] = self.df_result[self.date_column].astype('category')
        
        # self.df_result[self.date_column] = self.df_result[self.date_column].apply(lambda x: datetime.strptime(x, "%y-%m-%d").strftime("%d/%m/%y"))
    container = st.container()
    if self.date_column == 'False':
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
    
    if self.date_column != 'False':
    # Crear el gráfico de Altair con líneas de umbrales y sombreado
        chart = alt.Chart(self.df_result).mark_point().encode(
            x=f'{x_column}:N',
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


    
    
    if self.date_column == 'False': 
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
    
    if self.date_column == 'False':
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

    # st.write('HOLA')
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




