import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(path):
    return pd.read_csv(path)

housing = load_data("housing.csv")

#Mostramos el principio del data frama housing
print("Housing dataframe head\n")
print(housing.head())

#Mostramos stadisticas de cada atributo del data frama housing
print("Statistics:\n")
print(housing.describe())

#Mostramos histogramas de los distintos atributos del data frama housing
housing.hist(bins=50, figsize=(20,15))
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)
plt.show()

#Calculamos el coeficiente de Person [1,-1] de todos los atributos con el del valor medio
#de las casas
#cuando mas cerca estan de 1 o -1 mayor es la correlacion lineal
#    --> 1 la recta de la regresión tiene una pendiente positiva
#    --> -1 la recta de la regresión tiene una pendiente negativa
corr_matrix = housing.corr()
print(
    corr_matrix["median_house_value"].sort_values(ascending=False)
)
