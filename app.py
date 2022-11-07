import streamlit as st
import pandas as pd 
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pickle


anime_info = pd.read_csv('./data/anime_tv.csv')
pivot_table = pd.read_hdf('./data/pivot_table.h5')
model = pickle.load(open('knnpickle_file', 'rb'))

with st.sidebar: 
    choice = st.radio("Navigation", ["Anime Search","Anime Recommendation"])

if choice == 'Anime Search':
    types = list(anime_info['Type'].drop_duplicates())
    types.insert(0, 'Any')
    prem = list(anime_info['Premiered'].drop_duplicates())
    prem.insert(0, 'Any')
    source = list(anime_info['Source'].drop_duplicates())
    source.insert(0, 'Any')
    studio = list(anime_info['Studios'].drop_duplicates())
    studio.insert(0, 'Any')

    anime_type = st.selectbox('Select anime / ova:', types)
    anime_premier = st.selectbox('Select premier season/ year', prem)
    anime_source = st.selectbox('Select source material', source)
    anime_studio = st.selectbox('Select anime studio', studio)

    if st.button('Search for Anime'):
        df_copy = anime_info.copy()
        df_copy.drop(columns=['Genres','Aired', 'Premiered', 'Duration', 'Ranked'], inplace=True)
        if anime_type != 'Any':
            df_copy = df_copy[df_copy['Type']== anime_type]
        if anime_premier != 'Any':
            df_copy = df_copy[df_copy['Premiered']== anime_premier]
        if anime_source != 'Any':
            df_copy = df_copy[df_copy['Source']== anime_source]
        if anime_studio != 'Any':
            df_copy = df_copy[df_copy['Studios']== anime_studio]

        st.dataframe(df_copy, 800, 300)

def predict(anime):
    indexes = list(pivot_table.index)
    anime_index = indexes.index(anime)
    query = pivot_table.iloc[anime_index, :].values.reshape(1, -1)
    distance, suggestions = model.kneighbors(query, n_neighbors=6)
    result = []
    for i in range(0, len(distance.flatten())):
        if i == 0:
            result.append('Recommendations for {0}:\n'.format(pivot_table.index[anime_index]))
        else:
            result.append('{0}: {1}'.format(i, pivot_table.index[suggestions.flatten()[i]]))
    return result

if choice == 'Anime Recommendation':
    st.title("Anime Recommendation")
    anime_recomm = st.selectbox('Select an Anime', list(pivot_table.index))
    if st.button('Get Recommendations'):
        predictions = predict(anime_recomm)
        for prediction in predictions:
            st.markdown(prediction)


