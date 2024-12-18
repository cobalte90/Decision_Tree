import pandas as pd
import sklearn.model_selection
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
import graphviz
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Создаём классификатор с критерием "энтропия"
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Импортируем датасет
df = pd.read_csv('train.csv')

# Делим датасет на фичи и таргет, а также на тренировочную и тестовую часть
y_train = df['Survived']
y_test = y_train.iloc[790:891]
y_train = y_train.drop([int(x) for x in range(790, 891)])

X_train = df.drop(['Survived', 'Cabin', 'Name', 'Ticket', 'Embarked'], axis=1)
X_train.Sex.replace(['male', 'female'], [0, 1], inplace=True)
X_train['Age'] = X_train['Age'].fillna(int(X_train['Age'].mean()))
X_test = X_train.iloc[790:891]
X_train = X_train.drop([int(x) for x in range(790, 891)])

# Обучаем классификатор с разными значениями параметра максимальной глубины дерева и проводим кросс-валидацию
CV_scores = pd.DataFrame({'max_depth' : [int(x) for x in range(1, 20)], 'CV_score' : [int(x) for x in range(1, 20)]})
for max_depth in range(1, 20):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    CV_scores.loc[max_depth-1,'CV_score'] = cross_val_score(clf, X_train, y_train, cv=10).mean()

# Находим лучшее значение глубины (с максимальным cross-validation score)
best_depth = CV_scores.query('CV_score == CV_score.max()')['max_depth'].iloc[0]

# Обучаем классификатор с наилучшим параметром глубины дерева
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
clf.fit(X_train, y_train)
print('The best tree depth: ', best_depth, '\nThe best cross-validation score:', round(cross_val_score(clf, X_test, y_test).mean(), 5))


# Импротируем данные для предсказания, приводим их к нужному виду
New_X_test = pd.read_csv('test.csv')
New_X_test = New_X_test.drop(['Cabin', 'Embarked', 'Ticket', 'Name'], axis=1)
New_X_test['Sex'] = New_X_test['Sex'].replace(['male', 'female'], [0,1])

# Делаем предсказание, записываем его в csv-таблицу
pred = clf.predict(New_X_test)
pred = pd.DataFrame({'PassengerId' : [int(x) for x in range(892, 1310)],'Survived' : pred})
pred.to_csv('predicted_values.csv', index=False)

# Выводим в консоль предсказанные значения
p = pd.read_csv('predicted_values.csv')
print(p)

# Визуализируем наше дерево решений с помощью graphviz
data_graph = tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns, class_names=['Not_Survived', 'Survived'], filled=True)
graph = graphviz.Source(data_graph, format='png')
graph.render('decision_tree')
# Визуализацию можно увидеть в файле "decision_tree.png"


