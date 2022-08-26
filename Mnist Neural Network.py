#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist


# In[2]:


#get train and test sets and normalize the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


# In[3]:


model = tf.keras.models.Sequential()

#add layer to flatten the input
model.add(tf.keras.layers.Flatten())
#time for hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#only 10 possible classifications so only 10 outputs and softmax for probability
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# In[4]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5 )


# In[5]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[6]:


pred = model.predict([x_test])

print(np.argmax(pred[0]))


# In[ ]:




