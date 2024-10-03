import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Cargar el archivo CSV
df = pd.read_csv('german_credit.csv')

# Verificar el balance original de clases
original_counts = df["Creditability"].value_counts()

# Separar las clases mayoritaria y minoritaria
df_majority = df[df.Creditability == 1]
df_minority = df[df.Creditability == 0]

# Calcular la media armónica entre el tamaño de las clases
harmonic_mean = int(2 * len(df_minority) * len(df_majority) / (len(df_minority) + len(df_majority)))

# Sobremuestrear la clase minoritaria
df_minority_upsampled = resample(df_minority, 
                                 replace=True,  # Muestreo con reemplazo
                                 n_samples=harmonic_mean,  # Ajustar el tamaño según la media armónica
                                 random_state=42)

# Combinar las clases mayoritaria y sobremuestreada
df_resampled = pd.concat([df_majority, df_minority_upsampled])

# Verificar el nuevo balance de clases
resampled_counts = df_resampled["Creditability"].value_counts()

# Crear gráficos de barras de conteo con los datos originales y ajustados
plt.figure(figsize=(12, 6))

# Gráfico de las clases originales
plt.subplot(1, 2, 1)
sns.barplot(x=original_counts.index, y=original_counts.values, palette='viridis')
plt.title('Balance Original de Clases')
plt.xlabel('Clase de Crédito')
plt.ylabel('Conteo')
plt.xticks([0, 1], ['No Aprobado (0)', 'Aprobado (1)'])

# Gráfico de las clases después del balanceo
plt.subplot(1, 2, 2)
sns.barplot(x=resampled_counts.index, y=resampled_counts.values, palette='viridis')
plt.title('Balance de Clases Después del Sobremuestreo')
plt.xlabel('Clase de Crédito')
plt.ylabel('Conteo')
plt.xticks([0, 1], ['No Aprobado (0)', 'Aprobado (1)'])

# Ajustar el layout y mostrar los gráficos
plt.tight_layout()
plt.show()

# Guardar el nuevo DataFrame en un archivo CSV
df_resampled.to_csv('german_credit_resampled.csv', index=False)
