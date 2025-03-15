def cargar_datos():
    data_path = '../data'
    files = os.listdir(data_path)
    
    df_list = []

    for file in files:
        if file.endswith('.csv'):  
            file_path = os.path.join(data_path, file)

            parts = file.split('_')

            if len(parts) < 3:
                continue
            
            year = parts[0]  
            month_name = parts[2]  

            if year not in ['2023', '2024']:
                continue

            if month_name not in ['Gener', 'Febrer', 'Marc', 'Abril', 'Maig']:
                continue

            try:
                df = pd.read_csv(file_path, low_memory=True, dtype=str, skiprows=lambda i: i % 1000 != 0)

                df['last_reported'] = pd.to_datetime(df['last_reported'], unit='s', errors='coerce')

                df['year'] = df['last_reported'].dt.year
                df['month'] = df['last_reported'].dt.month
                df['day'] = df['last_reported'].dt.day
                df['hour'] = df['last_reported'].dt.hour

                if 'traffic' in df.columns:
                    df.drop(columns=['traffic'], inplace=True)
                
                if 'V1' in df.columns:
                    df.drop(columns=['V1'], inplace=True)

                # Convertir columnas a float
                for col in ['num_bikes_available','num_docks_available','num_bikes_available_types.mechanical', 'num_bikes_available_types.ebike']:
                    df[col] = df[col].astype('float64')

                df_list.append(df)

            except Exception:
                continue

    merged_df = pd.concat(df_list, ignore_index=True) if df_list else None

    if merged_df is not None:
        df_2 = pd.read_csv('../data/Informacio_Estacions_Bicing_2025.csv', usecols=['station_id', 'lat', 'lon','capacity'], low_memory=False)
        
        merged_df['station_id'] = merged_df['station_id'].astype(str)
        df_2['station_id'] = df_2['station_id'].astype(str)
        
        merged_df = merged_df.merge(df_2, on='station_id', how='left')

    # Crear columna sum_capacity
    merged_df['sum_capacity'] = merged_df['num_bikes_available'] + merged_df['num_docks_available']

    # Calcular la mediana de sum_capacity por estación
    median_capacity = merged_df.groupby('station_id')['sum_capacity'].mean()

    def impute_capacity(row):
        return median_capacity[row['station_id']] if pd.isna(row['capacity']) else row['capacity']

    merged_df['capacity'] = merged_df.apply(impute_capacity, axis=1)
    merged_df['diff_capacity_available'] = merged_df['capacity'] - (merged_df['num_bikes_available'] + merged_df['num_docks_available'])

    merged_df.loc[merged_df['diff_capacity_available'] > 0, 'num_docks_available'] += merged_df['diff_capacity_available']
    merged_df.loc[merged_df['diff_capacity_available'] < 0, 'num_docks_available'] += merged_df['diff_capacity_available']

    # Asegurar límites de num_docks_available
    merged_df['num_docks_available'] = merged_df.apply(
        lambda row: min(max(row['num_docks_available'], 0), row['capacity']),
        axis=1
    )

    # Crear columna target (% de bicis disponibles)
    merged_df['target'] = merged_df['num_docks_available'] / merged_df['capacity']

    # **Agrupar a nivel de hora por estación**
    aggregated_df = merged_df.groupby(['station_id', 'year', 'month', 'day', 'hour']).agg(
        num_bikes_available=('num_bikes_available', 'mean'),
        num_docks_available=('num_docks_available', 'mean'),
        num_mechanical=('num_bikes_available_types.mechanical', 'median'),
        num_ebike=('num_bikes_available_types.ebike', 'median'),
        target=('target', 'mean'),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        capacity=('capacity', 'first')
    ).reset_index()

    id = pd.read_csv('../data/metadata_sample_submission_2025.csv')

    llista_stations = pd.unique(id['station_id'])

    aggregated_df = aggregated_df['station_id'].isin(llista_stations)

    return aggregated_df
      
