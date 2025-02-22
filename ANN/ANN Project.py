'''
Lending Club Project
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,explained_variance_score

# Function to fill missing mort_acc based on total_acc
def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

# Import data
data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
df = pd.read_csv('lending_club_loan_two.csv')

# Data Preprocessing and Analysis
df = df.drop('emp_title',axis=1)
df = df.drop('emp_length',axis=1)
df = df.drop('title',axis=1)

total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)

# Convert term to int
df['term'] = df['term'].apply(lambda term: int(term[:3]))

# One-hot encoding
df = pd.get_dummies(df, columns=['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type'], drop_first=True)
df = df.drop('grade',axis=1)
# Extract zip code from address
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df = df.drop('address',axis=1)
df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)

# Extract year from earliest_cr_line
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

df = df.drop('issue_d',axis=1)

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df = df.drop('loan_status',axis=1)
df = df.dropna()
# Training and Testing Data
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the Model
callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model = Sequential()
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Model
model.fit(x=X_train, y=y_train, batch_size=256, epochs=100, validation_data=(X_test, y_test), callbacks=[callback])

# Save Model
model.save('my_model.h5')

# Predictions
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(explained_variance_score(y_test, predictions))

