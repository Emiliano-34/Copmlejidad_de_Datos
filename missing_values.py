import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('missing_values_test.csv')
print("Dataset original:")
print(df)

numeric_columns = df.select_dtypes(include=['float64']).columns

# Imputar valores faltantes utilizando la media
imputer = SimpleImputer(strategy='mean')  # Se puede usar mean, median, most_frequent
df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)

df[numeric_columns] = df_imputed

print("\nDataset después de la imputación:")
print(df)

# Guardar el nuevo DataFrame en un archivo CSV
df.to_csv('missing_values_imputed.csv', index=False)
