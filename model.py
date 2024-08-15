import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv('C:/Users/ss/OneDrive/Desktop/Books/BR/kindle_data-v2.csv')  # Update with your actual path

# Basic preprocessing: Drop rows with missing values
data = data.dropna()

# Convert categorical variables into numeric
label_encoder_author = LabelEncoder()
label_encoder_soldBy = LabelEncoder()
label_encoder_category_name = LabelEncoder()

data['author'] = label_encoder_author.fit_transform(data['author'])
data['soldBy'] = label_encoder_soldBy.fit_transform(data['soldBy'])
data['category_name'] = label_encoder_category_name.fit_transform(data['category_name'])

# Convert binary columns to integers
data['isKindleUnlimited'] = data['isKindleUnlimited'].astype(int)
data['isBestSeller'] = data['isBestSeller'].astype(int)
data['isEditorsPick'] = data['isEditorsPick'].astype(int)
data['isGoodReadsChoice'] = data['isGoodReadsChoice'].astype(int)

# Convert 'publishedDate' to numeric (Year)
data['publishedDate'] = pd.to_datetime(data['publishedDate'], errors='coerce').dt.year
data = data.dropna(subset=['publishedDate'])
data['publishedDate'] = data['publishedDate'].astype(int)

# Select features and the target variable
X = data[['author', 'soldBy', 'price', 'isKindleUnlimited', 'category_id', 'isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'publishedDate', 'category_name']]
y = data['stars']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=100)

# Fit the model
xg_reg.fit(X_train, y_train)

# Predict on the test set to evaluate performance
preds = xg_reg.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("Mean Squared Error: %f" % mse)

# Save the model
model_filename = 'xgboost_book_recommendation_model.h5'
joblib.dump(xg_reg, model_filename)
print(f"Model saved to {model_filename}")
