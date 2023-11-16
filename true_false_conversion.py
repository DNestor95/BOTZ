import pandas as pd


def convert_bool_values_to_int(df):
    df['Verified'] = df['Verified'].astype(int)
    return df

file_path = 'bot_detection_data_copy.csv'

df = pd.read_csv('bot_detection_data_copy.csv')

df = convert_bool_values_to_int(df)

df.to_csv('modded_detection_data.csv', index = False)

