# %% [markdown]
# Rupam Das _
# Task 1 _
# Classification Of Iris Data set _

# %%
#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
df=pd.read_csv("Iris.csv")

# %%
df.head() #returns first 5 entries

# %%
df.head(10)

# %%
df.tail() #returns last 5 entries

# %%
df.shape

# %%
df.isnull().sum()

# %%
df.dtypes

# %%
data=df.groupby('Species')

# %%
data.head()

# %%
df['Species'].unique()

# %%
df.info()

# %% [markdown]
# Data Set Visualization

# %%
plt.boxplot(df['SepalLengthCm'])

# %%
df['SepalLengthCm'].hist()

# %%
df['SepalWidthCm'].hist()

# %%
df['PetalLengthCm'].hist()

# %%
df['PetalWidthCm'].hist()

# %%
#VISUALIZING THE WHOLE DATASET
sns.pairplot(df, hue='Species')

# %%
df.drop('Id',axis=1,inplace=True)

# %%
sp={'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}

# %%
df.Species=[sp[i] for i in df.Species]

# %%
df

# %%
X=df.iloc[:,0:4]

# %%
X

# %%
y = df.iloc[:,4]

# %%
y

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

# %%
#TRAINING MODELS
model=LinearRegression()

# %%
model.fit(X,y)

# %%
model.score(X,y)

# %%
model.coef_

# %%
model.intercept_

# %% [markdown]
# Making Prediction

# %%

y_pred=model.predict(X_test)

# %%
import numpy as np
print("Mean Squared error: %2f" % np.mean((y_pred - y_test) ** 2))


