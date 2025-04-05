import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Layer, BatchNormalization
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "watson_healthcare_modified-Cleaned.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Encode categorical columns
categorical_columns = ['Department', 'Gender', 'OverTime', 'JobRole', 'MaritalStatus', 'Education', 'Shift']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Selecting specified features
selected_features = [
    'Department', 'Age', 'Gender', 'OverTime', 'JobRole', 'DistanceFromHome',
    'JobSatisfaction', 'Education', 'MaritalStatus', 'PercentSalaryHike',
    'WorkLifeBalance', 'Shift', 'TotalWorkingYears', 'TrainingTimesLastYear', 'JobLevel'
]

X = df[selected_features]
y = df['Attrition']

# Normalize numerical features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Reshape for LSTM input
X = np.expand_dims(X, axis=1)
y = np.array(y)

# Compute class weights
class_counts = Counter(y)
total_samples = len(y)
class_weight_dict = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Attention Layer
class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(1,), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Define the BiLSTM model
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.005)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.005)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Attention Layer
    x = Attention()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# Build and compile model
model = build_model((1, X.shape[2]))
optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,
                    callbacks=[lr_callback, early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Feature Prediction Analysis
feature_impacts = []
original_pred = model.predict(X_test[:1])[0, 0]  # Get initial prediction for one sample

for i in range(X_test.shape[2]):
    X_perturbed = np.copy(X_test[:1])
    X_perturbed[0, 0, i] += np.std(X[:, 0, i]) * 0.1  # Perturb feature by 10% of its std
    new_pred = model.predict(X_perturbed)[0, 0]
    feature_impacts.append(new_pred - original_pred)  # Difference in prediction

# Plot Feature Prediction Contribution
def plot_feature_predictions(features, impacts):
    sorted_indices = np.argsort(impacts)
    plt.figure(figsize=(10, 5))
    plt.barh(np.array(features)[sorted_indices], np.array(impacts)[sorted_indices], color='skyblue')
    plt.axvline(0, color='gray', linestyle='dashed')
    plt.xlabel("Change in Attrition Prediction")
    plt.ylabel("Feature")
    plt.title("Feature Prediction Impact on Attrition - BiLSTM")
    plt.show()

plot_feature_predictions(selected_features, feature_impacts)

