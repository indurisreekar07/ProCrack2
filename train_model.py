import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("tasks_dataset.csv")

print(data.head())

X = data[[
    'Difficulty(1-10)',
    'Available_Hours_Per_Day',
    'Days_Left',
    'Distraction_Level(1-10)'
]]

y = data['Estimated_Completion_Days']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)


task = [[
    8,  
    3,  
    7,  
    6   
]]

predicted_days = model.predict(task)

print("\nPredicted Completion Time:")
print(round(predicted_days[0], 2), "days")

hours_per_day = 3

print("\nRecommended Plan:")
print("Work", hours_per_day, "hours/day")
print("Expected completion in", round(predicted_days[0], 1), "days")
