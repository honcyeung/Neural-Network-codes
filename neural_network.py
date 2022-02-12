#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf




get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')




np.dot(price, unit)




epsilon = 1e-15
y_preds = [1, 1, 0, 0, 1]
y_preds_new = [max(i, epsilon) for i in y_preds]
y_preds_new




y_preds_new = [min(i, 1-epsilon) for i in y_preds_new]
y_preds_new




y_preds_new = np.array(y_preds_new)
np.log(y_preds_new)




np.log(y_preds)




data = pd.read_csv('homeprices_banglore.csv')
data.head()




data.info()




def scaling(column):
    return column/max(column)




scaling(data.area)




from sklearn import preprocessing

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

scaled_X = sx.fit_transform(data.drop('price', axis = 1))
scaled_X




scaled_y = sy.fit_transform(np.asarray(data.price).reshape(data.shape[0], 1))
scaled_y




w = np.ones(shape = scaled_X.shape[1])
np.dot(w, scaled_X.T)




def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    
    number_of_features = X.shape[1]
    w = np.ones(shape = number_of_features)
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):
        y_predicted = np.dot(w, X.T)+b
        
        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        w = w - learning_rate*w_grad
        b = b - learning_rate*b_grad
        
        cost = np.mean(np.square(y_true-y_predicted)) # MSE
        
        if i%10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list




w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0]), 500)
w, b, cost




plt.plot(epoch_list, cost_list)
plt.show()




def prediction(area, bedroom, w, b):
    scaled_X = sx.transform([[area, bedroom]])[0]
    scaled_price = w[0]*scaled_X[0]+w[1]*scaled_X[1]+b
    return sy.inverse_transform([[scaled_price]])[0][0]
    
prediction(2600, 4, w, b)




prediction(1000, 2, w, b)




prediction(1500, 3, w, b)




from random import randint

def stochastic_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape = number_of_features)
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):
        random_index = randint(0, total_samples-1)
        sample_X = X[random_index]
        sample_y = y_true[random_index]
        y_predicted = np.dot(w, sample_X.T)+b
        
        w_grad = -(2/total_samples)*(sample_X.T.dot(sample_y-y_predicted))
        b_grad = -(2/total_samples)*(sample_y-y_predicted)
        w = w - learning_rate*w_grad
        b = b - learning_rate*b_grad
        
        cost = np.square(sample_y-y_predicted) # MSE
        
        if i%10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
            
    return w, b, cost, cost_list, epoch_list




w, b, cost, cost_list, epoch_list = stochastic_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0]), 10000)
w, b, cost




plt.plot(epoch_list, cost_list);




def mini_batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape = number_of_features)
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):


# Exercise: GPU performance for fashion mnist dataset
# This notebook is derived from a tensorflow tutorial here: https://www.tensorflow.org/tutorials/keras/classification So please refer to it before starting work on this exercise
# 
# You need to write code wherever you see your code goes here comment. You are going to do image classification for fashion mnist dataset and then you will benchmark the performance of GPU vs CPU for 1 hidden layer and then for 5 hidden layers. You will eventually fill out this table with your performance benchmark numbers



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
plt.imshow(train_images[0])
train_labels[0]
class_names[train_labels[0]]
plt.figure(figsize=(3,3))
for i in range(5):
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
    plt.show()
train_images_scaled = train_images / 255.0
test_images_scaled = test_images / 255.0




input_shape = train_images_scaled.shape[-2:]
input_shape




def get_model(hidden_layers=1):
    layers = []
    # Your code goes here-----------START
    # Create Flatten input layers
    # Create hidden layers that are equal to hidden_layers argument in this function
    # Create output 
    # Your code goes here-----------END
    
    input_shape = train_images_scaled.shape[-2:]
    layers.append(keras.layers.Flatten(input_shape = input_shape))
    
    for i in range(1, hidden_layers+1):
        layers.append(keras.layers.Dense(int(input_shape[0]/(i*2)), activation = 'relu'))
    layers.append(keras.layers.Dense(10, activation = 'sigmoid'))
    model = keras.Sequential(layers)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
model = get_model(1)
model.fit(train_images_scaled, train_labels, epochs=5)
model.predict(test_images_scaled)[2]
test_labels[2]
tf.config.experimental.list_physical_devices() 




# 5 Epochs performance comparison for 1 hidden layer
%%timeit -n1 -r1
with tf.device('/CPU:0'):
    model = get_model(1)
    model.fit(train_images_scaled, train_labels, epochs=5)




get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/GPU:0'):\n     model = get_model(1)\n    model.fit(train_images_scaled, train_labels, epochs=5)")




# 5 Epocs performance comparison with 5 hidden layers
%%timeit -n1 -r1
with tf.device('/CPU:0'):
    # your code here




get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/GPU:0'):\n    # your code here")




df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()




df.info()




df1 = df.copy()
df1.drop('customerID', axis = 1, inplace = True)
df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors = 'coerce')




df1.dropna(inplace = True)




df1.info()




df1.TotalCharges.hist(bins = 20);




sns.histplot(df1, x = 'tenure', hue = 'Churn', multiple = 'dodge');




def print_unique_columns(data):

    column_list = list(set(data.columns)-set(['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']))

    for i in column_list:
        print(f'{i}: {data[i].unique()}\n count: {data[i].unique().shape[0]}')

print_unique_columns(df1)




df1.replace('No phone service', 'No', inplace = True)
df1.replace('No internet service', 'No', inplace = True)




yes_no_columns = ['Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace = True)




df1.replace({'Female': 1, 'Male': 0}, inplace = True)




col = ['InternetService', 'Contract', 'PaymentMethod']

df2 = pd.get_dummies(data = df1, columns = col)
df2.columns




df2.sample(5)




print_unique_columns(df2)




df2.dtypes




cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df2[cols_to_scale].sample(3)




X = df2.drop('Churn', axis = 1)
y = df2.Churn




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)




X_train.shape, X_test.shape, y_train.shape, y_test.shape




from sklearn.metrics import classification_report

def ANN(X_train, X_test, y_train, y_test, loss, weights):
    model = keras.Sequential([
    keras.layers.Dense(26, input_dim = 26, activation = 'relu'),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
    if weights == -1:
        model.fit(X_train, y_train, epochs = 100)
    else:
        model.fit(X_train, y_train, epochs = 100, class_weight = weights)
    model.evaluate(X_test, y_test)
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    print(classification_report(y_test, y_preds))
    
    return y_preds




y_preds = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)




class_0, class_1 = df2.Churn.value_counts()
df_class_0 = df2[df2.Churn == 0]
df_class_1 = df2[df2.Churn == 1]




df_class_0.shape, df_class_1.shape


# # method 1: undersampling



df_class_0_under = df_class_0.sample(class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis = 0)
df_test_under.shape




df_test_under.Churn.value_counts()




X = df_test_under.drop('Churn', axis = 1)
y = df_test_under.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 15)




y_preds = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)




class_0, class_1


# # method 2: over sampling (blind copying)



df_class_1_over = df_class_1.sample(class_0, replace = True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis = 0)




X = df_test_under.drop('Churn', axis = 1)
y = df_test_under.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 15)




y_test.value_counts()




y_preds = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)


# # method 3: oversampling (SMOTE)



from imblearn.over_sampling import SMOTE

X = df_test_under.drop('Churn', axis = 1)
y = df_test_under.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 15)

smote = SMOTE(sampling_strategy = 'minority')
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)




y_train_sm.value_counts()




y_preds = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)


# # method 4: ensemble method with undersampling



X = df2.drop('Churn', axis = 1)
y = df2.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 15)




df3 = X_train.copy()
df3['Churn'] = y_train




df3_class_0 = df3[df3.Churn == 0]
df3_class_1 = df3[df3.Churn == 1]




df3_class_0.shape, df3_class_1.shape




def get_train_batch(df_majority, df_minority, start, end):

    df_train = pd.concat([df_majority[start:end], df_minority], axis = 0)
    X_train = df_train.drop('Churn', axis = 1)
    y_train = df_train.Churn
    
    return X_train, y_train
X_train, y_train = get_train_batch(df3_class_0, df3_class_1, 0, 1495)




y_pred1 = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)




X_train, y_train = get_train_batch(df3_class_0, df3_class_1, 1495, 2990)
y_pred2 = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)
X_train, y_train = get_train_batch(df3_class_0, df3_class_1, 2990, 4130)
y_pred3 = ANN(X_train, X_test, y_train, y_test, 'binary_crossentropy', -1)




y_pred_final = y_pred1.copy()

for i in range(len(y_pred_final)):
    n_ones = y_pred1[i]+y_pred2[i]+y_pred3[i]
    if n_ones>1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0




print(classification_report(y_test, y_pred_final))




data = pd.read_csv('sonar.csv', header = None)
data.head()




data.info()




data.isna().sum().unique()




data[60].value_counts()




X = data.drop(60, axis = 1)
y = data[60]




y = pd.get_dummies(y, drop_first = True)
y




y.value_counts()




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)




X_train.shape, X_test.shape




model = keras.Sequential([
    keras.layers.Dense(60, input_dim = 60, activation = 'relu'),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 100, batch_size = 8)




model.evaluate(X_test, y_test)




y_pred = model.predict(X_test).reshape(-1)
y_pred = np.round(y_pred)
y_pred[:10]




y_test[:10].T




from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))




model = keras.Sequential([
    keras.layers.Dense(60, input_dim = 60, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 100, batch_size = 8)




model.evaluate(X_test, y_test)




def mini_batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):

    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features. In 
    # our case number_of_features = 2 (area, bedroom)
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0] # number of rows in X
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):        
        batch_index = int(number_of_features/epochs)
        y_predicted = np.dot(w, X.T) + b

        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.mean(np.square(y_true-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_X,scaled_y.reshape(scaled_y.shape[0],),500)
w, b, cost




from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Churn_Modelling.csv')
data.head()




data.info()


# # 1. undersampling



class0_count, class1_count = data.Exited.value_counts()
data_class0 = data[data.Exited == 0]
data_class1 = data[data.Exited == 1]




class0_count, class1_count




data_class0_under = data_class0.sample(class1_count)
data_under = pd.concat([data_class0_under, data_class1], axis = 0).sample(frac = 1)
data_under.head()




data_under.Exited.value_counts()


# # 2. oversampling



data_class1_over = data_class1.sample(class0_count, replace = True)
data_over = pd.concat([data_class1_over, data_class0], axis = 0).sample(frac = 1)
data_over.head()




data_over.Exited.value_counts()


# # 3. oversampling (SMOTE)



from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split




X = data.drop('Exited', axis = 1)
y = data.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, 
                                                   random_state = 42)




smote = SMOTE(sampling_strategy = 'minority')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# # 4. ensemble method



X = data.drop('Exited', axis = 1)
y = data.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42,
                                                   test_size = 0.2)
df3 = X_train.copy()
df3['Exited'] = y_train

df3_class0 = df3[df3.Exited == 0]
df3_class1 = df3[df3.Exited == 1]

def get_training_batch(df_minority, df_majoirty, start end):
    df = pd.concat([df_minoirty[start:end], df_majority], axis = 0)
    X_train = df.drop('Exited', axis = 1)
    y_train = df.Exited
    
    return X_train, y_train
X_train, y_train = get_training_batch(df3_class1, df3_class0, 0, df3_class1.shape[0])
model = LogisticRegression().fit(X_train, y_train)
y_pred1 = model.predict(X_test)
X_train, y_train = get_training_batch(df3_class1, df3_class0, df3_class1.shape[0], df3_class1.shape[0]*2)
model = LogisticRegression().fit(X_train, y_train)
y_pred2 = model.predict(X_test)
X_train, y_train = get_training_batch(df3_class1, df3_class0, df3_class1.shape[0]*2, df3_class0.shape[0])
model = LogisticRegression().fit(X_train, y_train)
y_pred3 = model.predict(X_test)

y_pred_final = y_pred1.copy()

for i in range(len(y_pred1)):
    final = y_pred1[i]+y_pred2[i]+y_pred3[i]
    if final>1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0
        




from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding




reviews = ['nice food',
        'amazing restaurant',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement']

sentiment = np.array([1,1,1,1,1,0,0,0,0,0])




one_hot('amazing restaurant', 30)




vocab_size = 30
encoded_review = [one_hot(d, vocab_size) for d in reviews]
print(encoded_review)




max_length = 3
padded_review = pad_sequences(encoded_review, maxlen = max_length, padding = 'post')
print(padded_review)




embeded_vector_size = 4

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length = max_length, name = 'embedding'))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))




X = padded_review
y = sentiment




model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()




model.fit(X, y, epochs = 20, verbose = 0)




model.evaluate(X, y)




weights = model.get_layer('embedding').get_weights()[0]
len(weights)




import gensim

df = pd.read_json('reviews_Cell_Phones_and_Accessories_5.json', lines = True)
df.head()




df.shape




df.info()




df.reviewText[0]




review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
review_text




model = gensim.models.Word2Vec(
    window = 10, min_count = 2, workers = 4
)




model.build_vocab(review_text, progress_per = 1000)




model.epochs




model.corpus_count




model.train(review_text, total_examples = model.corpus_count, epochs = model.epochs)




model.save('word2vec_amazon_review.model')




model.wv.most_similar('bad')




model.wv.similarity(w1 = 'cheap', w2 = 'inexpensive')




model.wv.similarity(w1 = 'great', w2 = 'good')




model.wv.similarity(w1 = 'great', w2 = 'iphone')




import tensorflow as tf




daily_sales_numbers = [21, 22, -108, 31, -1, 32, 34,31]
tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)
tf_dataset




for sales in tf_dataset.as_numpy_iterator():
    print(sales)




for sales in tf_dataset.take(3):
    print(sales.numpy())




tf_dataset = tf_dataset.filter(lambda x: x>0)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)




tf_dataset = tf_dataset.map(lambda x: x*72)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)




tf_dataset = tf_dataset.shuffle(4)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)




for sales in tf_dataset.batch(4):
    print(sales.numpy())




tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)
tf_dataset = tf_dataset.filter(lambda x: x>0).map(lambda y: y*72).shuffle(2).batch(2)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)




image_ds = tf.data.Dataset.list_files('images/*/*', shuffle = False)
image_ds = image_ds.shuffle(200)

for file in image_ds.take(3):
    print(file.numpy())




class_names = ['cat', 'dog']
image_count = len(image_ds)
train_size = int(image_count)*0.8
train_ds = image_ds.take(train_size)
test_ds = image_ds.skip(train_size)




import os

def get_label(file_path):
    return tf.strings.split(file_path, os.path.sep)[-2]




def process_image(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    
    return img, label




for img, label in train_ds.map(process_img).take(3):
    print(img)
    print(label)




def scale(image, label):
    return image/255, label




import time

tf.__version__




class FileDataset(tf.data.Dataset):
    def read_files(num_samples):
        time.sleep(0.03)
        for sample_idx in range(num_samples):
            time.sleep(0.015)
            yield (sample_idx,)
    
    def __new__(cls, num_samples = 3):
        return tf.data.Dataset.from_generator(
        cls.read_files, output_signature = tf.TensorSpec(shape = (1, ), dtype = tf.int64),
        args = (num_samples,)
        )




def benchmark(dataset, num_epochs = 2):
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)




get_ipython().run_cell_magic('timeit', '', 'benchmark(FileDataset())')




get_ipython().run_cell_magic('timeit', '', 'benchmark(FileDataset().prefetch(1))')




get_ipython().run_cell_magic('timeit', '', 'benchmark(FileDataset().prefetch(tf.data.AUTOTUNE))')




dataset = tf.data.Dataset.range(5)
for d in dataset:
    print(d.numpy())




dataset = dataset.map(lambda x: x**2)
for d in dataset:
    print(d.numpy())




dataset = dataset.cache()




list(dataset)




def mapped_function(s):
    tf.py_function(lambda: time.sleep(0.03), [], ()) 
    return s




get_ipython().run_cell_magic('timeit', '-n1 -r1', '\nbenchmark(FileDataset().map(mapped_function), 5)')






