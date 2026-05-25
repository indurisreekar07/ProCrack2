import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("tasks_dataset.csv")

# Show first 5 rows
print("Dataset Preview:")
print(data.head())

# Features (inputs)
X = data[[
    'Difficulty(1-10)',
    'Available_Hours_Per_Day',
    'Days_Left',
    'Distraction_Level(1-10)'
]]

# Target (output)
y = data['Estimated_Completion_Days']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Example prediction
task = [[
    8,  # Difficulty
    3,  # Available Hours Per Day
    7,  # Days Left
    6   # Distraction Level
]]

# Predict completion days
predicted_days = model.predict(task)

print("\nPredicted Completion Time:")
print(round(predicted_days[0], 2), "days")

# Smart schedule logic
hours_per_day = 3

print("\nRecommended Plan:")
print("Work", hours_per_day, "hours/day")
print("Expected completion in", round(predicted_days[0], 1), "days")