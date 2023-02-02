import pandas as pd
import seaborn as sb
import sklearn as sk
df = pd.read_csv('insurance.csv')
df.head()
sb.boxplot(df['bmi'])
newdf = df.replace({'southwest':0, 'southeast':1, 'northwest':2, 'northeast':3, 'male':0,'female':1,'no':0,'yes':1})
newdf.head()
X = newdf[['age','sex','children','bmi','smoker','region']]
y = newdf[['charges']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train,y_train)
reg.score(X_train,y_train)
pred = reg.predict([['45','0','0','22','1','1']])
print(pred)