import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


#https://www.kaggle.com/datasets/camnugent/california-housing-prices
data = pd.read_csv('./housing.csv')
data.dropna(inplace=True)



X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

#Get a 20% of dataset to test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = X_train.join(y_train)

train_data.hist(figsize=(15, 8))
plt.show()


train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)


print(train_data)
train_data.hist(figsize=(15, 8))
plt.show()

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')

plt.show()

plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude', y='longitude', data=train_data, hue="median_house_value", palette="rocket")

plt.show()