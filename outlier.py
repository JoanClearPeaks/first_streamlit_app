import pandas as pd

# Lee el archivo CSV
predictive = pd.read_csv('predictive_maintenance.csv')
st.sidebar.subheader('COLUMN', help="Select the target column.")
target_column = st.sidebar.selectbox("Select the target column", [None] +  list(selection.columns), index=0, label_visibility="collapsed")


