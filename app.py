from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pre-saved data and matrices
user_items_matrix = joblib.load('user_items_matrix.pkl')
similarities = joblib.load('similarity_matrix.pkl')
df1 = pd.read_csv("user_interactions.csv")

# Function to find similar users (using pre-loaded similarity matrix)
def find_similar_user(user_id, user_item_matrix, n_users=5):
    similarities_df = pd.DataFrame(similarities, index=user_item_matrix.index, columns=user_item_matrix.index)
    similar_users = similarities_df[user_id].sort_values(ascending=False).index[1:n_users + 1]
    return similar_users

def recommend_movies_with_titles(user_id, user_item_matrix, original_df, n_recom=5):
    if user_id not in user_item_matrix.index:
        return f"Error: User ID {user_id} not found in the system."
    
    similar_users = find_similar_user(user_id, user_item_matrix)
    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings == 0].index
    similar_users_ratings = user_item_matrix.loc[similar_users, unseen_movies]
    recommended_scores = similar_users_ratings.mean(axis=0).sort_values(ascending=False).head(n_recom)
    recommended_df = original_df[original_df['show_id'].isin(recommended_scores.index)]
    recommended_df = recommended_df.drop_duplicates(subset='show_id')
    recommended_df = recommended_df.set_index('show_id').loc[recommended_scores.index]
    recommended_df['score'] = recommended_scores.values
    return recommended_df[['title', 'type', 'rating', 'score']]



@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error_message = None

    if request.method == 'POST':
        user_id = request.form['user_id']
        try:
            result = recommend_movies_with_titles(user_id, user_items_matrix, df1)
            if isinstance(result, str):
                error_message = result
            else:
                recommendations = result
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"

    return render_template('index.html', recommendations=recommendations, error_message=error_message)



@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
