import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load the dataset and prepare the necessary data
train_data = pd.read_csv('BigBasket Products 4500.csv')

# Fill missing ratings with 0 and drop other rows with missing values
train_data['rating'].fillna(0, inplace=True)
train_data.dropna(inplace=True)

# Create a combined feature for text similarity
train_data['combined_feature'] = train_data['description'] + ' ' + train_data['category'] + ' ' + train_data['sub_category'] + ' ' + train_data['type']

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['combined_feature'])

# Convert to sparse matrix
tfidf_matrix = csr_matrix(tfidf_matrix)

# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the recommendation function
def get_recommendations(product_name, top_n=5):
    idx = train_data[train_data['product'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return train_data.iloc[sim_indices][['product', 'category', 'sub_category', 'brand', 'sale_price', 'rating']]

# Streamlit app setup
st.title('BigBasket Recommender System')

# Create a selectbox for selecting a product
selected_product_name = st.selectbox(
    'Select a product:',
    train_data['product'].values
)

# When the recommend button is clicked, show the recommendations
if st.button('Recommend'):
    recommendations = get_recommendations(selected_product_name)
    st.write("Recommended Products:")
    st.write(recommendations)
