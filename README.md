# CAPSTONE PROJECT
---
#  Predicción de la disponibilidad de bicis en estaciones de Bicing de Barcelona

Este proyecto realiza la preparación y modelado de datos históricos del sistema Bicing en Barcelona para predecir la ocupación de estaciones. Se utilizan técnicas de procesamiento distribuido, enriquecimiento con variables externas (clima y tiempo), ingeniería de características y codificación temporal.

---

##  Librerías utilizadas

```python
import os  
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import gc
```

## Inicialización del entorno distribuido
```python
def inicialize_dask():
    client = Client(memory_limit='8GB', processes=False)
    print(client)
```
Se inicializa un cliente Dask para distribuir la carga de trabajo y mejorar el rendimiento en la lectura de datos masivos.


## Carga y selección de archivos relevantes
```python
def cargar_datos():
    inicialize_dask()
    data_path = '../data'
    files = os.listdir(data_path)
    selected_files = []

    for file in files:
        if file.endswith('.csv'):
            parts = file.split('_')
            if len(parts) < 3:
                continue
            year, month = parts[0], parts[1]
            if year not in ['2024','2023','2022]:
                continue
            if month not in ['06','07','08','09','10','11','12']:
                continue
            selected_files.append(os.path.join(data_path, file))
```
Filtrado de archivos por año y mes. Se mantienen solo los años 2022, 2023 y 2024.

## Procesamiento paralelo de archivos CSV
```python
    @delayed
    def process_file(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=True, dtype=str, 
                usecols=[
                    'station_id', 'last_reported', 'num_bikes_available',
                    'num_docks_available', 'num_bikes_available_types.mechanical',
                    'num_bikes_available_types.ebike', 'is_installed', 'is_renting',
                    'is_returning', 'is_charging_station'
                ],
                skiprows=lambda i: i > 0 and i % 3 != 0)
            
            df['last_reported'] = pd.to_datetime(df['last_reported'], unit='s', errors='coerce')
            df['year'] = df['last_reported'].dt.year
            df['month'] = df['last_reported'].dt.month
            df['day'] = df['last_reported'].dt.day
            df['hour'] = df['last_reported'].dt.hour

            numeric_cols = [
                'num_bikes_available', 'num_docks_available',
                'num_bikes_available_types.mechanical', 'num_bikes_available_types.ebike'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            df = df.dropna(how='any')
            return df
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()
```
Se extraen columnas relevantes, se transforman tipos y se eliminan registros nulos. El procesamiento es distribuido con Dask para mejorar el rendimiento.

## Fusión con metadatos de estaciones y creación de variable objetivo
```python
    delayed_dfs = [process_file(file_path) for file_path in selected_files]
    if not delayed_dfs:
        return None

    ddf = dd.from_delayed(delayed_dfs)
    df_meta = pd.read_csv('../Informacio_Estacions_Bicing_2025.csv',
                          usecols=['station_id', 'lat', 'lon', 'capacity'],
                          low_memory=False)

    ddf['station_id'] = ddf['station_id'].astype('Int64')
    df_meta['station_id'] = df_meta['station_id'].astype('Int64')

    for col in ['is_installed', 'is_renting', 'is_returning']:
        if col in ddf.columns:
            ddf[col] = ddf[col].astype(str).replace({'nan': '0', '<NA>': '0'}).astype(int)

    if 'is_charging_station' in ddf.columns:
        ddf['is_charging_station'] = ddf['is_charging_station'].astype(str).map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    ddf = ddf.dropna(how='any')
    ddf = ddf.merge(df_meta, on='station_id', how='inner')
    ddf['sum_capacity'] = ddf['num_bikes_available'] + ddf['num_docks_available']
    df_final = ddf.compute()

    median_capacity = df_final.groupby('station_id')['sum_capacity'].median()
    df_final['capacity'] = df_final['capacity'].fillna(df_final['station_id'].map(median_capacity))
    df_final['num_docks_available'] = df_final['num_docks_available'].clip(lower=0, upper=df_final['capacity'])
    df_final['target'] = df_final['num_docks_available'] / df_final['capacity']
```
Se imputan valores nulos de varias variables y se calcula la variable objetivo 'target' como el porcentaje de disponibilidad de docks. 


## Agregación horaria y filtrado por estaciones activas
```python
    aggregated_df = df_final.groupby(['station_id', 'year', 'month', 'day', 'hour']).agg(
        num_bikes_available=('num_bikes_available', 'mean'),
        num_docks_available=('num_docks_available', 'mean'),
        num_mechanical=('num_bikes_available_types.mechanical', 'median'),
        num_ebike=('num_bikes_available_types.ebike', 'median'),
        is_renting=('is_renting', 'mean'),
        is_returning=('is_returning', 'mean'),
        target=('target', 'mean'),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        capacity=('capacity', 'first')
    ).reset_index()

    id_df = pd.read_csv('../data/metadata_sample_submission_2025.csv')
    station_list = pd.unique(id_df['station_id'])
    aggregated_df = aggregated_df[aggregated_df['station_id'].isin(station_list)]

    return aggregated_df
```
Se genera un dataset agregado por hora con estadísticas por estación y se filtran las estaciones que forman parte del conjunto de predicción.


## Variables de contexto (lags)
```python
def crear_campos_optimized(df):
    df = df.sort_values(by=['station_id', 'year', 'month', 'day', 'hour']).reset_index(drop=True)

    for lag in range(1, 5):
        df[f'ctx-{lag}'] = df.groupby('station_id')['target'].shift(lag)

    mask = df.groupby('station_id').cumcount() >= 4
    df = df[mask]
    df = df.iloc[::5].reset_index(drop=True)

    return df
```
Se crean 4 variables que representan los valores de las 4 horas anteriores de la variable target, y se filtran las observaciones sin historial suficiente.

##  Clasificación de tipo de día (laboral, fin de semana, festivo)
```python
def day_categorization_bcn(df):
    holiday_set = set([...])  # Lista de festivos 2020-2025

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_type'] = 0
    df.loc[df['date'].dt.weekday >= 5, 'day_type'] = 1
    df.loc[df['date'].astype(str).isin(holiday_set), 'day_type'] = 2

    return df.drop(columns=['date'])
```
Se añade una variable según el tipo del día (laborable, finde de semana y festivo)  para capturar patrones cíclicos o estructurales según la tipología del calendario.

## Variables meteorológicas (clima diario)
```python
def weather_features(df_merge_final):
    export = pd.read_csv('../export.csv', parse_dates=["date"])

    export["rainy_day"] = export["prcp"].apply(lambda x: 1 if x > 1 else 0)
    export["windy_day"] = export["wspd"].apply(lambda x: 1 if x > 30 else 0)
    export["hot_day"] = export["tmax"].apply(lambda x: 1 if x > 15 else 0)

    export["year"] = export["date"].dt.year
    export["month"] = export["date"].dt.month
    export["day"] = export["date"].dt.day

    df_final = df_merge_final.merge(export, on=["year", "month", "day"], how="inner")
    return df_final
```
Se integran variables binarias que reflejan condiciones climáticas por día.


## Codificación cíclica de mes, día y hora
```python
def create_cyclic_features(df):
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)

    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df
```
Codificación trigonométrica para representar relaciones temporales de forma continua, evitando saltos artificiales en el cambio de valores como "mes 12 a mes 1".

##  Preparación de Datos para Predicción

Este bloque ejecuta la carga completa de datos, aplica ingeniería de características y enriquece la información con variables temporales y climáticas.

---

```python
# Cargar y procesar los datos históricos desde múltiples archivos
df_merge = cargar_datos()

# Generar variables de contexto basadas en valores anteriores del objetivo (lags)
df_merge = crear_campos_optimized(df_merge)

# Clasificar cada observación como día laboral, fin de semana o festivo
df_merge_final = day_categorization_bcn(df_merge)

# Añadir variables climáticas diarias (lluvia, viento, temperatura)
df_merge_final = weather_features(df_merge_final)

# Codificar mes, día y hora como variables cíclicas (sin y cos)
df_merge_final = create_cyclic_features(df_merge_final)
```

##  Modelado Predictivo de Ocupación de Estaciones

Esta sección implementa diferentes modelos de regresión (Red Neuronal, Regresión Lineal y Random Forest) para predecir la ocupación de estaciones de Bicing.

---

##  Selección de variables de entrada

```python
X = df_merge_final[['station_id', 'month', 'day', 'hour', 'ctx-1', 'ctx-2', 'ctx-3', 'ctx-4', 
                    'lat', 'lon', 'day_type', 'rainy_day', 'windy_day', 'hot_day', 
                    'sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour']]
y = df_merge_final['target']
```

## Red Neuronal (MLPRegressor)
```python
def neural_network_model(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_size, random_state=42)

    percent_features = ['ctx-1', 'ctx-2', 'ctx-3', 'ctx-4']
    bounded_features = ['month', 'day', 'hour']
    continuous_features = ['lat', 'lon']
    categorical_features = ['station_id', 'day_type']
    weather_features = ['rainy_day', 'windy_day', 'hot_day']  
    cyclic_features = ['sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour']

    missing_cols = [col for col in percent_features + bounded_features + continuous_features +
                    categorical_features + weather_features + cyclic_features if col not in X_train.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in X: {missing_cols}")

    preprocessor = ColumnTransformer(transformers=[
        ('percent', MinMaxScaler(), percent_features),
        ('bounded', MinMaxScaler(), bounded_features),
        ('continuous', StandardScaler(), continuous_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('weather', MinMaxScaler(), weather_features),
        ('cyclic', StandardScaler(), cyclic_features)
    ])

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_validation_scaled = y_scaler.transform(y_validation.values.reshape(-1, 1))

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=128,
        n_iter_no_change=10,
        verbose=1
    )

    pipeline = Pipeline([('preprocess', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train_scaled.ravel())

    y_pred_scaled = pipeline.predict(X_validation)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    r2 = r2_score(y_validation_scaled, y_pred)
    mse = mean_squared_error(y_validation_scaled, y_pred)
    mae = mean_absolute_error(y_validation_scaled, y_pred)

    return r2, mse, mae, pipeline, X_validation, y_validation
```

## Regresión Lineal
```python
def linear_regression(X, y, size):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mse, mae
```

## Random Forest Regressor
```python
def RandomForest(X, y, size):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    feature_importance = rf.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return r2, mse, mae, importance_df, rf
```

## Evaluación de modelos
```python
# Regresión Lineal
r2_lr, mse_lr, mae_lr = linear_regression(X, y, size=0.3)
print(f"R² Linear Regression: {r2_lr}, MSE: {mse_lr}, MAE: {mae_lr}")

# Random Forest
r2_rf, mse_rf, mae_rf, importance_df_rf, rf_model = RandomForest(X, y, size=0.3)
print(f"R² Random Forest: {r2_rf}, MSE: {mse_rf}, MAE: {mae_rf}")
print("Importancia de variables (RF):")
print(importance_df_rf)

# Red Neuronal
r2_nn, mse_nn, mae_nn, nn_model, X_validation_data, y_validation_data = neural_network_model(X, y, 0.3)
print(f"R² NN: {r2_nn}, MSE: {mse_nn}, MAE: {mae_nn}")

```

## Comprativa de métricas
```python
metrics_df = pd.DataFrame({
    'Métrica': ['R²', 'MSE', 'MAE'],
    'Neural Network': [r2_nn, mse_nn, mae_nn],
    'Linear Regression': [r2_lr, mse_lr, mae_lr],
    'Random Forest': [r2_rf, mse_rf, mae_rf]
})
print(metrics_df)
```

#  Predicción Final y Generación del Archivo de Entrega

Este bloque realiza la preparación de los datos de predicción y genera las predicciones finales usando el modelo entrenado. Finalmente, se guarda el archivo `predictions.csv` con el formato requerido para la entrega.

---

##  Carga parcial del conjunto de predicción

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc

use_cols = ['index', 'station_id', 'month', 'day', 'hour', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1']
df = pd.read_csv('../data/metadata_sample_submission_2025.csv', usecols=use_cols)
df['year'] = 2024  # Requerido por funciones posteriores
```

## Imputación de coordenadas lat/lon
```python
df_merge_final = df_merge_final[['station_id', 'lat', 'lon']].drop_duplicates()
df = df.merge(df_merge_final, on='station_id', how='left')
```

## Clasificación de tipo de día
```python
df = day_categorization_bcn(df)
```

## Integración de variables climáticas
```python
df = weather_features(df)
```

## Codificación de variables categóricas
```python
if df['day_type'].dtype == 'object':
    df['day_type'] = LabelEncoder().fit_transform(df['day_type'])
```

## Generación de variables cíclicas
```python
df = create_cyclic_features(df)
```

## Definición de variables para predicción
```python
features = ['station_id', 'month', 'day', 'hour', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1', 
            'lat', 'lon', 'day_type', 'rainy_day', 'windy_day', 'hot_day', 
            'sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour']
X_predict = df[features]
```

## Predicción en lotes con modelo entrenado
```python
batch_size = 5000
predictions = []

for start in range(0, len(X_predict), batch_size):
    end = start + batch_size
    batch = X_predict.iloc[start:end]
    
    preds = nn_model.predict(batch)  # Modelo previamente entrenado
    predictions.extend(preds)

    del batch
    gc.collect()
```

## Creación del archivo final para entrega
```python
df['percentage_docks_available'] = predictions
df_final = df[['index', 'percentage_docks_available']]
df_final.to_csv('predictions.csv', index=False)
```




---
---




# Side Project: Variables Espaciales para Estaciones Bicing

Este análisis complementario estudia el contexto geoespacial de las estaciones de Bicing en Barcelona, integrando variables como centralidad en la red vial y proximidad a estaciones de metro.

---

##  Descarga y visualización de la red vial

```python
import osmnx as ox

G = ox.graph_from_place("Barcelona, Spain", network_type="drive")
fig, ax = ox.plot_graph(G)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
```

---

##  Cálculo de Betweenness Centrality

```python
import networkx as nx

G_nx = nx.Graph(G)
betweenness = {node: 0 for node in G_nx.nodes()}

for source in G_nx.nodes():
    shortest_paths = nx.single_source_shortest_path(G_nx, source, cutoff=5)
    for target, path in shortest_paths.items():
        if source != target:
            for node in path[1:-1]:
                betweenness[node] += 1
```

---

##  Visualización de betweenness en la red vial

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(5, 50))
scaled_sizes = scaler.fit_transform(np.array(list(betweenness.values())).reshape(-1, 1)).flatten()
node_color = [plt.cm.Reds(betweenness.get(node, 0)) for node in G_nx.nodes()]
node_size = [scaled_sizes[i] for i, node in enumerate(G_nx.nodes())]

fig, ax = ox.plot_graph(G, node_color=node_color, node_size=node_size,
                        edge_color="gray", bgcolor="white", show=False)
plt.suptitle("Node Betweenness Centrality in Road Network")
plt.title("Note: Darker colors indicate higher betweenness values.")
plt.savefig("betweenness_centrality.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

##  Asociación entre estaciones Bicing y betweenness
```python
from scipy.spatial import KDTree
import pandas as pd

bicing_stations = pd.read_csv("/content/Informacio_Estacions_Bicing_2025.csv")
node_positions = {node: (data['y'], data['x']) for node, data in G.nodes(data=True)}
node_coords = np.array(list(node_positions.values()))
node_ids = list(node_positions.keys())
tree = KDTree(node_coords)

radius = 1000 / 111320
bicing_stations["sum_betweenness"] = np.nan

for idx, row in bicing_stations.iterrows():
    station_coord = (row["lat"], row["lon"])
    nearby_node_indices = tree.query_ball_point(station_coord, r=radius)
    nearby_nodes = [node_ids[i] for i in nearby_node_indices]
    sum_betweenness = np.sum([betweenness.get(node, 0) for node in nearby_nodes])
    bicing_stations.at[idx, "sum_betweenness"] = sum_betweenness

bicing_stations.to_csv("bicing_with_sum_betweenness.csv", index=False)
```
Se suma la centralidad de los nodos viales más cercanos (en un radio de 1 km) a cada estación de Bicing.

---

##  Visualización de estaciones Bicing con betweenness

```python
fig, ax = ox.plot_graph(G, show=False, edge_color="gray", bgcolor="white")
sc = ax.scatter(
    bicing_stations["lon"], bicing_stations["lat"],
    c=bicing_stations["sum_betweenness"], cmap="Reds",
    edgecolors="black", alpha=0.8, s=40
)
plt.colorbar(sc, label="Sum of Betweenness of Nodes within 1 Km")
plt.title("BICING Stations' Betweenness")
plt.savefig("bicing_betweenness.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

##  Cálculo de estaciones de metro cercanas
Se calcula cuántas estaciones de metro hay a 1000m de cada estación de Bicing.
```python
metro_df = pd.read_csv("/content/metro_stops.csv")
points_df = bicing_stations[['station_id', 'lat', 'lon']]

points_coords = np.array(points_df[["lat", "lon"]])
metro_coords = np.array(metro_df[["lat", "lon"]])
tree = KDTree(metro_coords)

radius = 1000 / 111320
points_df["metro_count"] = [len(tree.query_ball_point(coord, r=radius)) for coord in points_coords]
points_df.to_csv("points_with_metro_count.csv", index=False)
```

---

##  Variable dependiente: disponibilidad media de docks

```python
data_depvar = pd.read_csv('/content/data/2024_05_Maig_BicingNou_ESTACIONS.csv')
grouped_df = data_depvar[['station_id','num_docks_available']].groupby('station_id').mean()

bicing_station = pd.read_csv('/content/bicing_with_sum_betweenness.csv')
merged_df = pd.merge(bicing_station, grouped_df, on='station_id', how='inner')
merged_df_2 = pd.merge(merged_df, points_df, on='station_id', how='inner')

merged_df_2['target'] = merged_df_2['num_docks_available'] / merged_df_2['capacity'] * 100
```

---

##  Regresión OLS: betweenness, metro y localización

```python
import statsmodels.api as sm

X = np.log1p(merged_df_2['sum_betweenness'])
X = np.column_stack([X, np.log1p(merged_df_2['metro_count'])])
X = np.column_stack([X, merged_df_2['lat_x']])
X = np.column_stack([X, merged_df_2['lon_y']])
X = sm.add_constant(X)

Y = merged_df_2['target']
model = sm.OLS(Y, X).fit()
print(model.summary())
```
Se construye un modelo lineal para explicar la disponibilidad de anclajes en función de:

- Centralidad vial (log-transf)

- Densidad de metro (log-transf)

- Latitud y longitud
