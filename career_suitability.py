import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Training data (Math %, English %, Life Sciences %, Personality score)
X = np.array([
    [85, 70, 60, 9],  # High marks + high personality score
    [60, 65, 55, 6],
    [75, 80, 70, 8],
    [50, 60, 50, 5],
    [90, 85, 80, 10]
])

# Target: IT Suitability Score (numeric 0-100)
y = np.array([90, 60, 85, 50, 95])

# 2. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 3. Test prediction: new student
new_student = np.array([[78, 75, 65, 8]])  # Math, English, Life Sciences, Personality
predicted_score = model.predict(new_student)

print("=== Career Suitability Prediction ===")
print(f"Math: {new_student[0][0]}, English: {new_student[0][1]}, Life Sciences: {new_student[0][2]}, Personality: {new_student[0][3]}")
print(f"Predicted IT Suitability Score: {predicted_score[0]:.2f}")
