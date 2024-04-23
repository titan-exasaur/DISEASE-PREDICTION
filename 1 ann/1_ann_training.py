# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# LOADING THE DATASET & PRE - PROCESSING
df = pd.read_csv('multi_disease.csv')

# Mapping target values
a = np.arange(0, len(df['prognosis'].unique()) + 1, 1).tolist()
b = df['prognosis'].unique().tolist()
c = {j: i for i, j in zip(a, b)}
d = {i:j for i,j in zip(a,b)}

print(d)

output = df['prognosis'].unique().tolist()

df['prognosis'] = df['prognosis'].map(c)
df = df.sample(frac=1)

# Dropping unnecessary feature
from sklearn.feature_selection import VarianceThreshold
var_thres = VarianceThreshold(threshold=0)
var_thres.fit(df)
df = df.drop(['fluid_overload'], axis=1)

# Splitting data
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1050, test_size=0.3)

# Defining a function to eliminate more features based on pearson coefficient
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(x_train, 0.45)  # 0.38 is the threshold value

# Dropping features based on correlation
x_train = x_train.drop(corr_features, axis=1)
x_test = x_test.drop(corr_features, axis=1)

# Dropping correlated features from the original raw_dataset
x_new = pd.concat([x_train,x_test])

# Saving the new raw_dataset to a CSV file
shortened_df = pd.concat([x_new, y], axis=1)  # Concatenate the features and target variable
shortened_df.to_csv('shortened_dataset.csv', index=False)

print("New raw_dataset with dropped columns saved successfully.")

# Importing TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Defining the ANN model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.2),  # Adding dropout for regularization
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(output), activation='softmax')  # Output layer with softmax activation for multiclass classification
])

# Compiling the model
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Defining ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint('ann_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

import matplotlib.pyplot as plt

# Training the ANN model with the ModelCheckpoint callback
history = ann_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])

# Plotting epoch vs accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Train Loss'], loc='upper left')
# plt.savefig('eval_1.png',bbox_inches='tight')
plt.show()

# Plotting epoch vs accuracy
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Val Accuracy', 'Val Loss'], loc='upper left')
# plt.savefig('eval_2.png',bbox_inches='tight')
plt.show()


# Evaluating the ANN model
ann_loss, ann_accuracy = ann_model.evaluate(x_test, y_test)
print("ANN Accuracy:", ann_accuracy)

# Predicting with the ANN model
ann_pred = np.argmax(ann_model.predict(x_test), axis=-1)

# Mapping encoded target values back to class labels
class_labels = [d[i] for i in range(len(d))]

# Printing class labels
print("Class Labels:", class_labels)

# Printing classification report with class labels
print('ANN Classification Report with Class Labels:\n')
print(classification_report(y_test, ann_pred, target_names=class_labels))


print('[info] trained model saved...')
