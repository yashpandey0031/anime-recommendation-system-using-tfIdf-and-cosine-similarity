#this is a workaround version for anime recommendation system , there is a lot better work with proper 
#commenting , will add it soon(main2.ipynb) ;/ , DO NOT EXCEED 100MB LIMIT THAT SHIT CAUSES ISSUES

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st



df=pd.read_csv("anime.csv") #reading the data
df.sample(5)

df.info()
# drop anime_id , type , episodes 
df.drop(columns=['anime_id','type','episodes'],inplace=True) #inplace = true makes the changes in the original object / data
df.isnull().sum() #check for missing values
df.dropna(subset=['genre', 'rating'], inplace=True) #check for missing values in genre and rating and drop them 
# check for duplicate values
df.duplicated().sum()

#we will keep the name as it is since it is a label
#Standardize separators (remove inconsistent spacing)
df['genre'] = df['genre'].str.replace(r'\s*,\s*', ', ', regex=True)
#Split genre strings into lists
df['genre'] = df['genre'].str.split(', ')
 #Strip whitespace from each genre (just in case)
df['genre'] = df['genre'].apply(lambda x: [g.strip() for g in x])
from collections import Counter

# Flatten all genre lists into one big list
all_genres = [genre for sublist in df['genre'] for genre in sublist]

# Count frequency of each genre
genre_counts = Counter(all_genres)

# Convert to a DataFrame for easier viewing (optional)
genre_freq_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

##we are tryna find the combinatios 
# Step 1: Convert genre lists to sorted, comma-separated strings
df['genre_combo'] = df['genre'].apply(lambda g: ', '.join(sorted(g)))

df['genre_processed'] = df['genre'].apply(lambda g: ' '.join(g).lower()) #removing the commas and lowercasing them 



# Initialize vectorizer
tfidf = TfidfVectorizer()

# Fit and transform the genre_processed column
genre_tfidf = tfidf.fit_transform(df['genre_processed'])



scaler = MinMaxScaler()
normalized_members = scaler.fit_transform(df[['members']])

normalized_rating = scaler.fit_transform(df[['rating']])


final_features = hstack([genre_tfidf, normalized_members, normalized_rating])



similarity_matrix = cosine_similarity(final_features)

def recommend(anime_name, df, similarity_matrix, top_n=5):
    # Get index of the anime
    idx = df[df['name'].str.lower() == anime_name.lower()].index
    if len(idx) == 0:
        return "Anime not found."
    idx = idx[0]

    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity (excluding the anime itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Get recommended anime names
    recommended = df.iloc[[i[0] for i in sim_scores]]['name'].tolist()
    return recommended

# --- Streamlit UI ---
# st.snow() (funnystuff )

st.image("anime.jpg")
st.title("Anime Recommender")


with st.form("recommend_form"):
    anime_name = st.text_input("Enter an anime name",placeholder="pls enter japanease name like Shingeki no Kyojin for attack on titan")
    submitted = st.form_submit_button("Recommend")


# with st.status("Searching for Anime..."):
#     st.write("Searching for data...")

# st.button("Rerun")


if submitted:
    if anime_name.strip() == "":
        st.warning("Please enter an anime name.")
    else:
        recommendations = recommend(anime_name, df, similarity_matrix)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.subheader("Recommended Anime:")
            for i, name in enumerate(recommendations, 1):
                st.write(f"{i}. {name}")

