import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('missing_values_test.csv')

# Remplaza los valores faltantes usando una estadistica descritiva (moda, mediana, media)
numeric_columns = df.select_dtypes(include=['float64']).columns
print(df)

imputer = SimpleImputer(strategy='mean')  #Se puede usar mean, median, most_frequent
df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_columns]))

print(df_imputed)



