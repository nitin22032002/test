import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("heart.csv")
# age

fig=plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', kde=True)
plt.ylabel('Total')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()

# sex(gender) 

labels=['Female', 'Male']
sns.countplot(data=df,x='sex')
plt.xlabel('Gender')
plt.ylabel('Total')
plt.xticks([0,1], labels)
plt.title('Sex(Gender) Distribution')
plt.show()

# chest pain 

labels=['Type 0', 'Type 2', 'Type 1', 'Type 3']
sns.countplot(data=df,x='cp')
plt.xlabel('Pain Type')
plt.ylabel('Total')
plt.xticks([0,1,2,3], labels)
plt.title('Chest Pain Type Distribution')
plt.show()

# resting blood pressure

sns.histplot(data=df, x='trestbps', kde=True)
plt.ylabel('Total')
plt.xlabel('resting blood pressure')
plt.title('resting blood pressure distribution')
plt.show()

# cholestrol

sns.histplot(data=df, x='chol', kde=True)
plt.ylabel('Total')
plt.xlabel('Cholestrol')
plt.title('Serum Cholestrol distribution')
plt.show()

# Fasting Blood Sugar

labels=['< 120 mg/dl', '> 120 mg/dl']
sns.countplot(data=df,x='fbs')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Total')
plt.xticks([0,1], labels)
plt.title('Fasting Blood Sugar Distribution')
plt.show()

# Resting Electrocardiographic 

sns.countplot(data=df,x='restecg')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Total')
plt.title('Resting Electrocardiographic Distribution')
plt.show()

# thalach (Maximum Heartrate)

sns.histplot(data=df, x='thalach', kde=True)
plt.ylabel('Total')
plt.xlabel('Maximum Heart Rate')
plt.title('Maximum Heart Rate Distribution')
plt.show()

# Exercise Induced Angina Distribution

labels=['False', 'True']
sns.countplot(data=df,x='exang')
plt.xlabel('Exercise Induced Angina Distribution')
plt.ylabel('Total')
plt.xticks([0,1], labels)
plt.title('Exercise Induced Angina Distribution')
plt.show()

# "oldpeak" Distribution

sns.histplot(data=df, x='oldpeak', kde=True)
plt.ylabel('Total')
plt.xlabel('oldpeak')
plt.title('"oldpeak" Distribution')
plt.show()

# slope distribution

sns.countplot(data=df,x='slope')
plt.xlabel('Slope')
plt.ylabel('Total')
plt.title('slope distribution')
plt.show()

# ca (Number of Major Vessels)

sns.countplot(data=df,x='ca')
plt.xlabel('Number of Major Vessels')
plt.ylabel('Total')
plt.title('ca (Number of Major Vessels)')
plt.show()

# "thal" distribution

sns.countplot(data=df,x='thal')
plt.xlabel('Number of "thal"')
plt.ylabel('Total')
plt.title('"thal" distribution')
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------
# Result

labels=['True', 'False']
order=df['target'].value_counts().index
plt.figure(figsize=(12,7))
plt.suptitle('Heart Disease Distribution')

# --- Pie Chart ---
plt.subplot(1, 2, 1)
plt.title('Pie Chart')
plt.pie(df['target'].value_counts(), labels=labels)
centre=plt.Circle((0, 0), 0.45, fc='white')
plt.gcf().gca().add_artist(centre)
# --- Histogram ---
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram')
sns.countplot(x='target', data=df,order=order)
plt.xlabel('Heart Disease Status')
plt.ylabel('Total')
plt.xticks([0, 1], labels)
plt.show()

