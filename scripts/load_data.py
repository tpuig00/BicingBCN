def cargar_datos():
    import os
    import pandas as pd  #aquesta part de moment la poso aquí però, potser hauria d'anar al main (ho poso per si algu no les carrega)
    data_path = './data'
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

            try:
                df = pd.read_csv(file_path, low_memory=False, dtype=str)

                df['last_reported'] = pd.to_datetime(df['last_reported'], unit='s', errors='coerce')
                df['last_updated'] = pd.to_datetime(df['last_updated'], unit='s', errors='coerce')

                df['Year'] = year
                df['Month'] = month_name

                df_list.append(df)

            except Exception:
                continue

      merged_df = pd.concat(df_list, ignore_index=True) if df_list else None

    if merged_df is not None:
        df_2 = pd.read_csv('Informacio_Estacions_Bicing_2025.csv', usecols=['station_id', 'lat', 'lon'], low_memory=False) #aqui faig el merge amb el segon dataset, només agafo les variables geoespacials
        merged_df = merged_df.merge(df_2, on='station_id', how='left')

    return merged_df

df = cargar_datos()
