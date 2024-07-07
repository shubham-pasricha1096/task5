import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\\C tutorials\\Internship\\task 5\\heart.csv')

# Display the head, tail, columns, and info of the dataframe
print(df.head())
print(df.tail())
print(df.columns.values)
print(df.isna().sum())
print(df.info())

# Plot histograms for the dataframe
df.hist(bins=50, grid=False, figsize=(20, 15))
plt.show()

# Describe the dataframe
print(df.describe())

# "1. How many people have heart disease and how many people doesn't have heart disease?"
print(df.target.value_counts())

df.target.value_counts().plot(kind='bar', color=["red", "yellow"])
plt.title("Heart Disease Values")
plt.xlabel("1 = Heart Disease , 0 = No Heart Disease")
plt.ylabel("Amount")
plt.show()

df.target.value_counts().plot(kind='pie', figsize=(8, 6))
plt.legend(['Disease', "No disease"])
plt.show()

# "2. People of which sex has most heart disease?"
df.sex.value_counts().plot(kind='pie', figsize=(8, 6))
plt.title('MaleFemale Ratio')
plt.legend(['Male', 'Female'])
plt.show()

print(pd.crosstab(df.target, df.sex)) 

sns.countplot(x='target', data=df, hue='sex')
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")
plt.show()

# "3. People of which sex has which type of chest pain most?"
df.cp.value_counts()

df.cp.value_counts().plot(kind='bar', color=['salmon', 'lightblue', 'seagreen', 'khaki'])
plt.title("Chest Pain Type vs Count")
plt.show()

pd.crosstab(df.sex, df.cp)

pd.crosstab(df.sex, df.cp).plot(kind='bar', color=['coral', 'lightblue', 'plum', 'khaki'])
plt.title("Type of Chest Pain for Sex")
plt.xlabel('0 = Female, 1 = Male')
plt.show()

# "4. People with which chest pain are more prone to have heart disease?"
pd.crosstab(df.cp, df.target)

sns.countplot(x='cp', data=df, hue='target')
plt.title("Chest Pain Type vs Heart Disease")
plt.show()


# 5. Average Age of People with Heart Disease vs. Without Heart Disease
avg_age_heart_disease = df[df['target'] == 1]['age'].mean()
avg_age_no_heart_disease = df[df['target'] == 0]['age'].mean()

plt.figure(figsize=(8, 6))
plt.bar(['Heart Disease', 'No Heart Disease'], [avg_age_heart_disease, avg_age_no_heart_disease], color=['salmon', 'orchid'])
plt.title('Average Age of People with and without Heart Disease')
plt.ylabel('Average Age')
plt.xlabel('Condition')
plt.show()

# 6. Distribution of Heart Disease Across Different Types of Resting Electrocardiographic Results (restecg)
restecg_distribution = pd.crosstab(df['restecg'], df['target'])

restecg_distribution.plot(kind='bar', stacked=True, figsize=(10, 6), color=['orchid', 'salmon'])
plt.title('Distribution of Heart Disease Across Different Types of Resting Electrocardiographic Results')
plt.xlabel('Resting Electrocardiographic Results (restecg)')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])
plt.show()

# 7. Correlation Between Exercise-Induced Angina (exang) and Heart Disease
exang_distribution = pd.crosstab(df['exang'], df['target'])

exang_distribution.plot(kind='bar', stacked=True, figsize=(10, 6), color=['orchid', 'salmon'])
plt.title('Correlation Between Exercise-Induced Angina and Heart Disease')
plt.xlabel('Exercise-Induced Angina (exang)')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])
plt.show()


sns.displot(x='age', data=df, bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.displot(x='thalach', data=df, bins=30, kde=True, color='chocolate')
plt.title("Thalach Distribution")
plt.show()
