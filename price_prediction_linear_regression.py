import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# Veri setini ekleme komutları
df = pd.read_csv("train.csv")

# Target ve feature ayırma
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Sayısal ve kategorik sütunları ayırma komutları
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Sayısal veriler
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Kategorik veriler
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Ön işleme
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Veri setini eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# LINEAR REGRESSION


linear_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

print("----- Linear Regression -----")
print("R2:", linear_r2)
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)



# RIDGE REGRESSION


ridge_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

print("\n----- Ridge Regression -----")
print("R2:", ridge_r2)
print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)



# LASSO REGRESSION


lasso_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", Lasso(alpha=0.1))
])

lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

lasso_r2 = r2_score(y_test, lasso_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

print("\n----- Lasso Regression -----")
print("R2:", lasso_r2)
print("MAE:", lasso_mae)
print("RMSE:", lasso_rmse)



# GRAFİKLER


# 1 SalePrice dağılımı
plt.figure()
plt.hist(df["SalePrice"], bins=50)
plt.title("SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()


# 2 Gerçek vs Tahmin
plt.figure()
plt.scatter(y_test, linear_pred)
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()


# 3 Residual Plot
residuals = y_test - linear_pred

plt.figure()
plt.scatter(linear_pred, residuals)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
