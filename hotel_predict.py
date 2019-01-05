import sys

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('ggplot')

path = r"C:\Users\miair\PycharmProjects\untitled3\reviews_train.csv" #путь к файлу reviews_train.csv
path_stars = r"C:\Users\miair\PycharmProjects\untitled3\hotels_train.csv" #путь к файлу hotels_train.csv
path_test = r"C:\Users\miair\PycharmProjects\untitled3\reviews_test.csv" #путь к файлу reviews_test.csv
path_predictions = r"C:\Users\miair\PycharmProjects\untitled3\hotels_test.csv" #путь к файлу hotels_test.csv
data = pd.read_csv(path, skip_blank_lines =True) #чтение reviews_train.csv
data_stars = pd.read_csv(path_stars, skip_blank_lines =True) #чтение hotels_train.csv
data_test = pd.read_csv(path_test, skip_blank_lines =True) #чтение reviews_test.csv
path_predictions = pd.read_csv(path_predictions, skip_blank_lines =True) #чтение hotels_test.csv
data['stars'] = data.merge(data_stars, 'left', on='hotel_id').stars #запись в dataframe data из dataframe data_stars оценок отелей с помощью ключа hotel_id

print('Choose algorithm text analysis(Bags-of-words or n-grams): ') #выбор алгоритма анализа текстовых признаков
Name_text_analysis = input() #название алгоритма анализа тестовых признаков
print('Choose ML algorithm(LogisticRegression, RandomForest, GBT or all): ') #выбор алгоритма машинного обучения
Name_algorithm_ML = input() #название алгоритма машинного обучения

stars_array = np.float_(data['stars'] * 10) #преобразование в массив оценок отелей
stars_array = np.int_(stars_array)
if (Name_text_analysis == 'Bags-of-words'):
    vectorizer = CountVectorizer().fit(data['text'])
    training_scores = vectorizer.transform(data['text']) #преобразование теста в массив числовых признаков с помощью алгоритма "мешок слов"
elif (Name_text_analysis == 'n-grams'):
    vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(data['text'])
    training_scores = vectorizer.transform(data['text']) #преобразование теста в массив числовых признаков с помощью алгоритма "n-grams"
else:
    print('Error')
    sys.exit()

if (Name_algorithm_ML == 'LogisticRegression'):
    model = LogisticRegression() #логистическая регрессия
    model.fit(training_scores, stars_array)
    predictions = model.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test = model.predict(vectorizer.transform(data_test['text'])) #предсказание тестового набора
elif(Name_algorithm_ML == 'RandomForest'):
    rf = ensemble.RandomForestClassifier() #Случайный лес
    rf.fit(training_scores, stars_array)
    predictions = rf.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test = rf.predict(vectorizer.transform(data_test['text'])) #предсказание тестового набора
elif(Name_algorithm_ML == 'GBT'):
    GBT = GradientBoostingClassifier() #Градиентный бустинг деревьев решений
    GBT.fit(training_scores, stars_array)
    predictions = GBT.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test = GBT.predict(vectorizer.transform(data_test['text'])) #предсказание тестового набора
elif(Name_algorithm_ML == 'all'): #все алгоритмы(LogisticRegression RandomForest GBT)
    model = LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    GBT = GradientBoostingClassifier()
    model.fit(training_scores, stars_array)
    rf.fit(training_scores, stars_array)
    GBT.fit(training_scores, stars_array)
    predictions_LR = model.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test_LR = model.predict(vectorizer.transform(data_test['text']))  # предсказание тестового набора
    predictions_RF = rf.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test_RF = rf.predict(vectorizer.transform(data_test['text']))  # предсказание тестового набора
    predictions_GBT = GBT.predict(vectorizer.transform(data['text'])) #предсказание обучающего набора
    predictions_test_GBT = GBT.predict(vectorizer.transform(data_test['text']))  # предсказание тестового набора
    print('LogisticRegression RNMSE: ', sqrt(mean_squared_error(stars_array / 10, predictions_LR / 10)))
    print('RandomForest RNMSE: ', sqrt(mean_squared_error(stars_array / 10, predictions_RF / 10)))
    print('GBT RNMSE: ', sqrt(mean_squared_error(stars_array / 10, predictions_GBT / 10)))
    print('Choose algorithm for predict data_test: ')
    Name_algorithm_predict = input()
    if (Name_algorithm_predict == 'LogisticRegression'):
        data['predict'] = predictions_LR / 10
        data_test['stars'] = predictions_test_LR / 10
    elif (Name_algorithm_predict == 'RandomForest'):
        data['predict'] = predictions_RF / 10
        data_test['stars'] = predictions_test_RF / 10
    elif (Name_algorithm_predict == 'GBT'):
        data['predict'] = predictions_GBT / 10
        data_test['stars'] = predictions_test_GBT / 10
else:
    print('Error')
    sys.exit()


if (Name_algorithm_ML != 'all'):
    data['predict'] = predictions / 10
    data_test['stars'] = predictions_test / 10
    stars_array = stars_array / 10
    predictions = predictions / 10
    print(Name_algorithm_ML, ' RNMSE: ', sqrt(mean_squared_error(stars_array, predictions))) #RMSE

path_predictions['stars'] = path_predictions.merge(data_test, 'left', on='hotel_id').stars  # запись в dataframe path_predictions из dataframe data_test оценок отелей с помощью ключа hotel_id
path_predictions.to_csv('hotels_test.csv', sep=',', index=False)  # запись в csv предсказаний оценок


