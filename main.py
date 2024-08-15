from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
model = joblib.load('C:/Users/ss/OneDrive/Desktop/Books/xgboost_book_recommendation_model.h5')

# Load the dataset
data = pd.read_csv('C:/Users/ss/OneDrive/Desktop/Books/BR/kindle_data-v2.csv')
data = data.dropna()

# Preprocess the data the same way as during training
label_encoder_author = LabelEncoder()
label_encoder_soldBy = LabelEncoder()
label_encoder_category_name = LabelEncoder()

data['author'] = label_encoder_author.fit_transform(data['author'])
data['soldBy'] = label_encoder_soldBy.fit_transform(data['soldBy'])
data['category_name'] = label_encoder_category_name.fit_transform(data['category_name'])

data['isKindleUnlimited'] = data['isKindleUnlimited'].astype(int)
data['isBestSeller'] = data['isBestSeller'].astype(int)
data['isEditorsPick'] = data['isEditorsPick'].astype(int)
data['isGoodReadsChoice'] = data['isGoodReadsChoice'].astype(int)

data['publishedDate'] = pd.to_datetime(data['publishedDate'], errors='coerce').dt.year
data = data.dropna(subset=['publishedDate'])
data['publishedDate'] = data['publishedDate'].astype(int)

# Recommendation function
def recommend_books(category_name, num_recommendations=5):
    category_id = label_encoder_category_name.transform([category_name])[0]
    filtered_books = data[data['category_name'] == category_id]

    if filtered_books.empty:
        return None

    filtered_books['Predicted-Rating'] = model.predict(filtered_books[['author', 'soldBy', 'price', 'isKindleUnlimited', 'category_id', 'isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'publishedDate', 'category_name']])
    top_books = filtered_books.sort_values(by='Predicted-Rating', ascending=False).head(num_recommendations)
    return top_books[['title', 'author', 'Predicted-Rating']]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    category_name = request.form.get('category_name')
    recommendations = recommend_books(category_name)

    if recommendations is not None:
        recommendations = recommendations.to_dict(orient='records')
        return render_template('results.html', recommendations=recommendations, category_name=category_name)
    else:
        return render_template('results.html', recommendations=[], category_name=category_name)

if __name__ == '__main__':
    app.run(debug=True)
