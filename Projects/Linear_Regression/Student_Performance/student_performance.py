import pandas as pd

df = pd.read_csv(
    r"D:\Coding\ML\Projects\Linear_Regression\Student_Performance\data\StudentPerformanceFactors.csv"
)

# print(df.head())
#print(df.info())
# print(df.describe())
#print(df.isnull().sum())


df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)
df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)
df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)

print(df.isnull().sum())

print(df.duplicated().sum())
df = df.drop_duplicates()

for col in df.select_dtypes(include="object"):
    print(f"\n{col}")
    print(df[col].unique())

df = pd.get_dummies(
    df,
    drop_first=True
)
print(df.info())

X = df.drop("Exam_Score", axis=1)

y = df["Exam_Score"]

from sklearn.model_selection import train_test_split

train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.2, 
    random_state=42
    ) 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully")

y_pred = model.predict(X_test)

print("First 5 actual scores:", y_test[:5])
print("First 5 model scores:", y_pred[:5])

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)

r2= r2_score(y_test, y_pred)

print("\nMean Squared Error: ", mse)
print("R2 percentage: ", r2*100)

print(X_train.shape)
print(X_test.shape)
print(df.shape)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef_df.sort_values(
    by="Coefficient",
    ascending=False
))


#Random Forest Comparison
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators= 100, random_state= 42)

model.fit(X_train, y_train)

print("Model Trained Successfully")

y_pred2 = model.predict(X_test)

print("First 5 actual scores:", y_test[:5])
print("First 5 model scores:", y_pred2[:5])

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred2)

r2= r2_score(y_test, y_pred2)

print("\nMean Squared Error: ", mse)
print("R2 percentage: ", r2*100)

import matplotlib.pyplot as plt

plt.scatter(y_pred2, y_pred)

plt.xlabel("Forest")

plt.ylabel("Linear")

plt.show()