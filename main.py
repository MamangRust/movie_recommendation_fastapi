import uvicorn
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI()

# Inisialisasi Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Membaca dataset
df = pd.read_csv("./dataset/movie_dataset.csv")
df = df.fillna('')

# Menggabungkan fitur menjadi satu string
def combine_features(row):
    return row['title'] + ' ' + row['genres'] + ' ' + row['director'] + ' ' + row['keywords'] + ' ' + row['cast']

df['combined_features'] = df.apply(combine_features, axis=1)

# Membuat CountVectorizer dan cosine similarity matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# Fungsi untuk mendapatkan rekomendasi film berdasarkan judul
def get_recommendations(title):
    movie_index = df[df['title'] == title].index[0]
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]
    recommended_movies = [df.iloc[i[0]]['title'] for i in sorted_similar_movies]
    return recommended_movies

# Fungsi untuk mendapatkan film dengan rating tertinggi berdasarkan genre
def get_top_movies_by_genre(genre, n=5):
    genre_df = df[df['genres'].str.contains(genre, case=False)]
    top_movies = genre_df.nlargest(n, 'vote_average')
    return top_movies['title'].tolist()

# Endpoint untuk rekomendasi film berdasarkan judul
@app.get("/recommend/", response_class=HTMLResponse)
async def recommend_movie(request: Request, title: str = None):
    """
    Get recommended movies based on the provided movie title.
    
    Parameters:
    - title (str, optional): The title of the movie. If not provided, a random movie will be recommended.

    Returns:
    - HTMLResponse: An HTML response containing recommended movies.
    """
    if title:
        recommended_movies = get_recommendations(title)
        movie_title = title
    else:
        random_movie_index = random.randint(0, len(df) - 1)
        random_movie_title = df.iloc[random_movie_index]['title']
        recommended_movies = get_recommendations(random_movie_title)
        movie_title = "Random Movie"
    return templates.TemplateResponse("recommendation.html", {"request": request, "movie_title": movie_title, "recommended_movies": recommended_movies})


# Endpoint untuk film dengan rating tertinggi berdasarkan genre
@app.get("/top_movies/", response_class=HTMLResponse)
async def top_movies_by_genre(request: Request, genre: str = "Action", n: int = 5):
    """
    Get top rated movies by genre.

    Parameters:
    - genre (str, optional): The genre of the movies. Default is "Action".
    - n (int, optional): The number of top rated movies to retrieve. Default is 5.

    Returns:
    - HTMLResponse: An HTML response containing top rated movies by genre.
    """
    top_movies = get_top_movies_by_genre(genre, n)
    return templates.TemplateResponse("top_movies.html", {"request": request, "genre": genre, "top_movies": top_movies})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
