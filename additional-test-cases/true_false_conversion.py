import pandas as pd

# function to convert boolean values to int values by checking the elements of 
# the 'Verified" column
def convert_bool_values_to_int(df): 
    # replaces original bool val to type int
    df['Verified'] = df['Verified'].astype(int)
    # returns the modifed data frame value 
    return df

# reads the .csv file into the data frame
df = pd.read_csv('bot_detection_data_copy.csv') 

# sends the data frame into the function 'convert_bool_values_to_int
df = convert_bool_values_to_int(df)  

# creates a new .csv file, 'modded_detection_data,' containing the data frame after being
# modified by convert_bool_values_to_int 
df.to_csv('modded_detection_data.csv', index = False)  

