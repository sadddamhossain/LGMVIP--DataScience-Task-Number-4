#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Iris.csv")

df


# # Data Analyze

# In[3]:


df.info()


# In[4]:


df.describe()


# In[6]:


print(df.shape)


# In[7]:


print(df.dtypes)


# In[8]:


df.nunique()


# In[10]:


print(df.columns)


# In[12]:


unique_values = df['SepalLengthCm'].unique()
unique_values


# In[14]:


unique_values = df['SepalWidthCm'].unique()
unique_values


# In[16]:


unique_values = df['PetalLengthCm'].unique()
unique_values


# In[17]:


unique_values = df['PetalWidthCm'].unique()
unique_values


# In[19]:


unique_values = df['Species'].unique()
unique_values


# In[25]:


count_value = df['Species'].value_counts()
count_setosa = count_value['Iris-setosa']
count_setosa


# In[34]:


count_value = df['Species'].value_counts()
count_versicolor = count_value['Iris-versicolor']
count_versicolor


# In[35]:


count_value = df['Species'].value_counts()
count_virginica = count_value['Iris-virginica']
count_virginica


# # Visualization

# In[30]:


import matplotlib.pyplot as plt
import seaborn as sn


# In[38]:


# Assuming you have counted the occurrences of each species
count_setosa = count_value['Iris-setosa']
count_versicolor = count_value['Iris-versicolor']
count_virginica = count_value['Iris-virginica']

# Create labels, sizes, and colors
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
sizes = [count_setosa, count_versicolor, count_virginica]
colors = ['blue', 'lightgray', 'red']
explode = (0.1, 0, 0)  # Explode the 'Iris-setosa' slice for emphasis

plt.figure(figsize=(8, 6))  # Set the figure size
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode)
plt.title('Distribution of Iris Species')
plt.show()


# In[39]:


# Pairplot (scatterplot matrix)
sns.pairplot(df, hue='Species', markers=["o", "s", "D"], palette="Set1")
plt.title('Pairplot of Iris Data by Species')


# In[40]:


# Box plot for each feature
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
sns.boxplot(x='Species', y='SepalLengthCm', data=df, palette="Set2")
plt.subplot(2, 3, 2)
sns.boxplot(x='Species', y='SepalWidthCm', data=df, palette="Set2")
plt.subplot(2, 3, 3)
sns.boxplot(x='Species', y='PetalLengthCm', data=df, palette="Set2")
plt.subplot(2, 3, 4)
sns.boxplot(x='Species', y='PetalWidthCm', data=df, palette="Set2")
plt.tight_layout()
plt.suptitle('Box Plots of Iris Data by Species', y=1.05)


# In[41]:


plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
sns.histplot(df, x='SepalLengthCm', hue='Species', kde=True, palette="Set3")
plt.subplot(2, 3, 2)
sns.histplot(df, x='SepalWidthCm', hue='Species', kde=True, palette="Set3")
plt.subplot(2, 3, 3)
sns.histplot(df, x='PetalLengthCm', hue='Species', kde=True, palette="Set3")
plt.subplot(2, 3, 4)
sns.histplot(df, x='PetalWidthCm', hue='Species', kde=True, palette="Set3")
plt.tight_layout()
plt.suptitle('Histograms of Iris Data by Species', y=1.05)


# In[43]:


# Filter the DataFrame to include only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Create a correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()


# # Decision tree 

# # model :-1

# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[59]:


X = df[['SepalLengthCm', 'PetalLengthCm']]
y = df['Species']


# In[60]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[61]:


# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)


# In[62]:


# Train the classifier on the training data
clf.fit(X_train, y_train)


# In[76]:


# Make predictions on the test data
y_pred = clf.predict(X_test)



# In[77]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[79]:


# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)


# # model :-2

# In[66]:


X = df[['SepalWidthCm', 'PetalWidthCm']]
y = df['Species']


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[87]:


clf = DecisionTreeClassifier(random_state=42)


# In[88]:


clf.fit(X_train, y_train)


# In[89]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[90]:


# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)


# In[ ]:




