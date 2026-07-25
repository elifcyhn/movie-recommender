import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

st.set_page_config(page_title="Film Öneri Motoru", page_icon="🍿", layout="wide")

@st.cache_data
def prepare_model():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    df = pd.merge(ratings, movies, on='movieId')
    df = df.drop('timestamp', axis=1)
    
    matrix = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(matrix)
    
    movies['genres'] = movies['genres'].str.replace('|', ', ')
    genre_dict = movies.drop_duplicates('title').set_index('title')['genres'].to_dict()
    
    return matrix, model, genre_dict

matrix, model_knn, genre_dict = prepare_model()


st.title("Akıllı Film Öneri Sistemi")
st.markdown("Makine öğrenmesi modeli, seçtiğiniz filme en yakın izleyici kitlesine sahip filmleri bulur.")
st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### ⚙️ Tercihlerinizi Belirleyin")
    film_list= sorted(matrix.index.tolist())
    selected_film = st.selectbox("Favori filminizi seçin:", film_list)
    
    recomm_count = st.slider("Kaç film önerilsin?", min_value=1, max_value=10, value=5)
    recomm_button = st.button("Filmleri Öner")

with right_col:
    if recomm_button:
        with st.spinner('Yapay zeka eşleşmeleri hesaplıyor...'):
            film_index = matrix.index.get_loc(selected_film)
            distances, indices = model_knn.kneighbors(matrix.iloc[film_index, :].values.reshape(1, -1), n_neighbors=recomm_count + 1)
            
            st.markdown("### 🏷️ Önerilen Filmlerin Detayları")
            
            for i in range(1, len(distances.flatten())):
                recomm_film = matrix.index[indices.flatten()[i]]
                similarity = (1 - distances.flatten()[i]) * 100 
                genre = genre_dict.get(recomm_film, "Bilinmiyor")
                
                st.info(f"**{recomm_film}** \n\n  **Tür:** {genre} | **Benzerlik:** %{similarity:.1f}")
 

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 17px;'> İyi seyirler dilerim!", unsafe_allow_html=True)