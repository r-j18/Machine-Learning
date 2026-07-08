import pandas as pd

df = pd.read_csv("Projects/Logistic_Regression/Customer_Churn/telecom_churn.csv")

df.head(5)

from sklearn.model_selection import train_test_split

X = df[['AccountWeeks', 'ContractRenewal']].values
y = df['Churn'].values

X_train, X_test, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)