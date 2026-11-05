Task 4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Remove unnecessary column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Predict new sales
new_data = pd.DataFrame([[200, 30, 40]],
                        columns=['TV', 'Radio', 'Newspaper'])
print("Predicted Sales:", model.predict(new_data)[0])