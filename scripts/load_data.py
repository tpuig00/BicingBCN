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

    return pd.concat(df_list, ignore_index=True) if df_list else None

df = cargar_datos()

