import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


df = pd.read_csv("bot_detection_data.csv")

print(df.head())


df  = df.dropna()

df['verified'] = df['verified'].astype(int)

x = df.drop(['Bot Label'], axis=1)
y= df['Bot Label']

## this seciton needs to have a feature created so that we can see if the post was interacted with by known bots 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

