mport pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from sklearn import tree

music_data = pd.read_csv('music.csv')

#dropping the genre column
x = music_data.drop(columns=['genre'])
y = music_data['genre']

# using 20% of dataset as sample testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x, y)

tree.export_graphviz(model,
                     out_file='music-recommended.dot',
                     feature_names=['age','gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)

#joblib.dump(model, 'music-recommended.joblib')
#model = joblib.load('music-recommended.joblib')

#predicitions
#predictions = model.predict([[21,1]])
#predictions

#calculating the accuracy of random prediction
#score = accuracy_score(y_test, predictions)
#score

music_data.describe()
music_data.values
