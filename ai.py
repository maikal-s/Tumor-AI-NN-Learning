import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('tumordata.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #Dropping the columns in the dataset that list whether a tumor is cancerous or not from X values.
y = dataset["diagnosis(1=m, 0=b)"] #Saving ONLY whether or not a tumor is cancerous as Y values.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #Setting aside 20% of the dataset to be used as test data to make sure the network is developing accurately

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid')) # Add hidden layers of neural network
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Add output layer, returning 0 or 1, depending on whether or not tumor is cancerous

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000) # Fits X to Y training data and runs over it 900 times. AI learns by repeatedly going over the same data.

model.evaluate(x_test, y_test) #Compares model training data to the test data to test for final accuracy