import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')
st.markdown("""
This app performs simple webscrapping of NBA player stats data!
* **Python libraries**: base64, pandas, streamlit
* **Data Source**: [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2022))))

# Web scrappuing or NBA player stats
@st.cache # to store cache data for the data that have been loaded before so we dont need to reload it

def loadData(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0) # fill NaN with 0
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = loadData(selected_year)

# sidebar - team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# sidebar - position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# filtering data based on input selection
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns')
st.dataframe(df_selected_team.astype(str))

# download nba player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def fileDownload(df):
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(fileDownload(df_selected_team), unsafe_allow_html = True)

# heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('basketball.csv', index = False)
    df = pd.read_csv('basketball.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,5))
        ac = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()