import tensorflow as tf
print(tf.__version__)

import numpy as np 
import pandas as pd
df = pd.read_csv('Lab  winequality-red .csv')
df.head()
y=df['quality']
x=df.drop('quality',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',max_iter=500)
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


df.head()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))


model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   #optimizer task - classification
#optimizer - Adam is an optimization algorithm that can be used instead of the classical stochastic
# gradient descent procedure to update network weights iteratively based on training data.
# It combines the advantages of two other extensions of stochastic gradient descent,
# namely Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
#loss - The loss function used here is 'sparse_categorical_crossentropy',
# which is suitable for multi-class classification problems where the target labels are integers.
# It calculates the cross-entropy loss between the true labels and the predicted probabilities.
#metrics - The model is evaluated using the 'accuracy' metric, which measures the proportion of correctly classified samples.
model.summary()
y_train=y_train-3
#why we are doing this because the quality of wine is in the range of 3-8 and we have to convert it into 0-5 for the model to work properly.
y_test=y_test-3 
#why we are doing this because the quality of wine is in the range of 3-8 and we have to convert it into 0-5 for the model to work properly.

print(np.unique(y_train))

model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))
#validation_split- 0.2 means 20% of the training data will be used for validation.
#epochs- The model will be trained for 100 epochs, meaning it will iterate over the entire training dataset 100 times.


pred = model.predict(x_test)
pred=np.argmax(pred,axis=1)
pred=pred+3
print(pred)


model.save('wine_quality_model.h5')

