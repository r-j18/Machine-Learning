'''Productivity Optimizer'''
'''Create a fake dataset using numpy containing hours studied, distracted hours and marks gained 
marks gained will be linearly related to study hours and distractions with some random noise
use pandas and matplotlib for data handling and visualisation and then
make an ML model using Linear Regression and find its accuracy and errors '''

'''Creating the dataset'''

import numpy as np

#Initialising Seed
rng = np.random.default_rng(seed = 42)

#Generating Study hours
min_study = 0
max_study = 11
sample_size = 100

study_hours = rng.uniform(low = min_study, high= max_study, size = sample_size)


#Generating Distracted Hours
min_distracted = 1
max_distracted = 11

distracted_hours = rng.uniform(low = min_distracted, high = max_distracted, size = sample_size)


#Generating noise for Score
mean = 0
std_dev = 2.0

noise = rng.normal(loc = mean, scale = std_dev, size = sample_size)

#Generating Score

study_slope = 8
distraction_slope = -3

raw_score = study_slope * study_hours + distraction_slope * distracted_hours + noise

score = np.maximum(raw_score, 0)

'''Using Pandas to structure the data'''

import pandas as pd

#Converting arrays into dictionary
sample_data = {
    'Study Hours' : study_hours,
    'Distracted Hours' : distracted_hours,
    'Noise' : noise,
    'Score' : score
}

#Converting Dictionary to Pandas DataFrame
df = pd.DataFrame(sample_data)

#Verify rows and column
# print(df.head())

'''Visualizing the data using Matplotlib'''

import matplotlib.pyplot as plt

#Setting up the canvas
fig, ax = plt.subplots(figsize = (9,6))

#Plot Points
ax.scatter(study_hours, score, color = 'teal', alpha = 0.7, edgecolors = 'black', label = 'Student Data')

#Calculate the trendline

cofefficients = np.polyfit(study_hours, score, deg = 1)
m_calc, b_calc = cofefficients

#Generate evenly spaced X values for a smooth line
x_line = np.linspace(min_study, max_study, 100)
y_line = m_calc * x_line + b_calc

ax.plot(x_line, y_line, color = 'crimson', linewidth = 2.5, linestyle = '--', label = f'Trendline (y = {m_calc:.2f}x + {b_calc:.2f})')

#Labling
ax.set_title('Study Hours(0-10) vs Exam Score', fontsize = 14, fontweight = 'bold', pad = 15 )
ax.set_xlabel('Study Hours (Independent Feature X)', fontsize = 11)
ax.set_ylabel('Exam Score (Target Label Y)', fontsize = 11)

#plt.show()

'''Training Model'''

#Isolating Matrices
X = df[['Study Hours', 'Distracted Hours']].values
y = df['Score'].values

#Verification
#print("Shape of X Matrix:", X.shape)
#print("Shape of Y vector:", y.shape)

#Split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

#Verification
# print("X_train shape: ", X_train.shape)
# print("X_test shape: ", X_test.shape)
# print("y_train shape: ", y_train.shape)
# print("y_test shape: ", y_test.shape)

#Importing and instantiating the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

#Train the model
model.fit(X_train, y_train)

#Verification
print("Model Training Successful")

#Prediction 
# y_prediction = model.predict(X_test)

#Compare model's prediction with real score
# print("First 5 Model Predictions:", y_prediction[:5])
# print("First 5 Actual Scores:", y_test[:5])

'''Evaluating the model'''

from sklearn.metrics import mean_squared_error, r2_score

#Calculate Mean Squared Error and R2 Score
# mse = mean_squared_error(y_test, y_prediction)
# r2 = r2_score(y_test, y_prediction)

# #Print the Evaluation
# print("====================================")
# print("     MODEL EVALUATION REPORT        ")
# print("====================================")
# print(f"Mean Squared Error (MSE) : {mse:.4f}")
# print(f"R-squared Score (R²)     : {r2:.4f} ({r2*100:.1f}%)")
# print("====================================")

print("Coefficients:", model.coef_)
print("Intercept:",model.intercept_)

'''Generate test data 1'''
rng_test = np.random.default_rng(seed = 42)
test_size = 50
study_hours_test = rng_test.uniform(low = min_study, high= max_study, size = test_size)

distracted_hours_test = rng_test.uniform(low = min_distracted, high = max_distracted, size = test_size)

new_std_dev = 2.0
noise_test = rng_test.normal(loc = mean, scale = new_std_dev, size = test_size)

raw_score_test = study_slope * study_hours_test + distraction_slope * distracted_hours_test + noise_test

score_test = np.clip(raw_score_test, 0, 100)

sample_data_test = {
    'Study Hours' : study_hours_test,
    'Distracted Hours' : distracted_hours_test,
    'Noise' : noise_test,
    'Score' : score_test
}

df_test = pd.DataFrame(sample_data_test)

#print(df_test.head())

X_new_test = df_test[['Study Hours', 'Distracted Hours']].values
y_new_test = df_test['Score'].values

y_prediction_test = model.predict(X_new_test)

print("First 5 Model Predictions:", y_prediction_test[:5])
print("First 5 Actual Scores:", y_new_test[:5])

#Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_new_test, y_prediction_test)
r2 = r2_score(y_new_test, y_prediction_test)


#Print the Evaluation
print("------------------------------------")
print("|    MODEL EVALUATION REPORT       |")
print("------------------------------------")
print(f"Mean Squared Error (MSE) : {mse:.4f}|")
print(f"R-squared Score (R²)     : {r2:.4f} ({r2*100:.1f}%)|")
print("------------------------------------")


'''Generate test data 2'''
rng_test = np.random.default_rng(seed = 0)
test_size = 50
study_hours_test = rng_test.uniform(low = min_study, high= max_study, size = test_size)

distracted_hours_test = rng_test.uniform(low = min_distracted, high = max_distracted, size = test_size)

new_std_dev = 2.0   
noise_test = rng_test.normal(loc = mean, scale = new_std_dev, size = test_size)

raw_score_test = study_slope * study_hours_test + distraction_slope * distracted_hours_test + noise_test

score_test = np.clip(raw_score_test, 0, 100)

sample_data_test = {
    'Study Hours' : study_hours_test,
    'Distracted Hours' : distracted_hours_test,
    'Noise' : noise_test,
    'Score' : score_test
}

df_test = pd.DataFrame(sample_data_test)

#print(df_test.head())

X_new_test = df_test[['Study Hours', 'Distracted Hours']].values
y_new_test = df_test['Score'].values

y_prediction_test = model.predict(X_new_test)

print("First 5 Model Predictions:", y_prediction_test[:5])
print("First 5 Actual Scores:", y_new_test[:5])

#Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_new_test, y_prediction_test)
r2 = r2_score(y_new_test, y_prediction_test)


#Print the Evaluation
print("------------------------------------")
print("|    MODEL EVALUATION REPORT       |")
print("------------------------------------")
print(f"Mean Squared Error (MSE) : {mse:.4f}|")
print(f"R-squared Score (R²)     : {r2:.4f} ({r2*100:.1f}%)|")
print("------------------------------------")

'''Generate test data 3'''
rng_test = np.random.default_rng(seed = 42)
test_size = 50
study_hours_test = rng_test.uniform(low = min_study, high= max_study, size = test_size)

distracted_hours_test = rng_test.uniform(low = min_distracted, high = max_distracted, size = test_size)

new_std_dev = 5.0
noise_test = rng_test.normal(loc = mean, scale = new_std_dev, size = test_size)

raw_score_test = study_slope * study_hours_test + distraction_slope * distracted_hours_test + noise_test

score_test = np.clip(raw_score_test, 0, 100)

sample_data_test = {
    'Study Hours' : study_hours_test,
    'Distracted Hours' : distracted_hours_test,
    'Noise' : noise_test,
    'Score' : score_test
}

df_test = pd.DataFrame(sample_data_test)

#print(df_test.head())

X_new_test = df_test[['Study Hours', 'Distracted Hours']].values
y_new_test = df_test['Score'].values

y_prediction_test = model.predict(X_new_test)

print("First 5 Model Predictions:", y_prediction_test[:5])
print("First 5 Actual Scores:", y_new_test[:5])

#Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_new_test, y_prediction_test)
r2 = r2_score(y_new_test, y_prediction_test)


#Print the Evaluation
print("------------------------------------")
print("|    MODEL EVALUATION REPORT       |")
print("------------------------------------")
print(f"Mean Squared Error (MSE) : {mse:.4f}|")
print(f"R-squared Score (R²)     : {r2:.4f} ({r2*100:.1f}%)|")
print("------------------------------------")

'''Generate test data 4'''
rng_test = np.random.default_rng(seed = 0)
test_size = 50
study_hours_test = rng_test.uniform(low = min_study, high= max_study, size = test_size)

distracted_hours_test = rng_test.uniform(low = min_distracted, high = max_distracted, size = test_size)

new_std_dev = 20.0
noise_test = rng_test.normal(loc = mean, scale = new_std_dev, size = test_size)

raw_score_test = study_slope * study_hours_test + distraction_slope * distracted_hours_test + noise_test

score_test = np.clip(raw_score_test, 0, 100)

sample_data_test = {
    'Study Hours' : study_hours_test,
    'Distracted Hours' : distracted_hours_test,
    'Noise' : noise_test,
    'Score' : score_test
}

df_test = pd.DataFrame(sample_data_test)

#print(df_test.head())

X_new_test = df_test[['Study Hours', 'Distracted Hours']].values
y_new_test = df_test['Score'].values

y_prediction_test = model.predict(X_new_test)

print("First 5 Model Predictions:", y_prediction_test[:5])
print("First 5 Actual Scores:", y_new_test[:5])

#Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_new_test, y_prediction_test)
r2 = r2_score(y_new_test, y_prediction_test)


#Print the Evaluation
print("------------------------------------")
print("|    MODEL EVALUATION REPORT       |")
print("------------------------------------")
print(f"Mean Squared Error (MSE) : {mse:.4f}|")
print(f"R-squared Score (R²)     : {r2:.4f} ({r2*100:.1f}%)|")
print("------------------------------------")

