import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\yogeshwaran\Downloads\chennai_house_price_dataset.csv")

X = df.drop("price", axis=1)
y = df["price"]

numeric_features = ["area", "bedrooms", "bathrooms", "floor", "age", "furnishing"]
categorical_features = ["location"]

preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_features),("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)])

model = Pipeline([("preprocessor", preprocessor),("regressor", RandomForestRegressor(n_estimators=300,random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("\nEnter House Details")

area = float(input("Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
floor = int(input("Floor: "))
age = int(input("Age (years): "))
furnishing = int(input("Furnishing (0=Unfurnished,1=Semi,2=Fully): "))
location = input("Location (Anna Nagar, Adyar, Velachery, OMR, Tambaram, Porur, T Nagar, Besant Nagar, Guindy, Chromepet): ")

new_house = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floor": floor,
    "age": age,
    "furnishing": furnishing,
    "location": location
}])

predicted_price = model.predict(new_house)

print("\nPredicted Price: ₹", round(predicted_price[0], 2))

avg_prices = df.groupby("location")["price"].mean().sort_values(ascending=False)

plt.figure()
avg_prices.plot(kind="bar")
plt.title("Average House Price by Location in Chennai")
plt.xlabel("Location")
plt.ylabel("Average Price (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
