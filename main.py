
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
import itertools

from mlxtend.data import iris_data

class LogReg:
    def __init__(self, learning_rate, n_epochs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, X, y):
        self.coef_ = np.random.normal(1, 0.5, size=X.shape[1])
        self.intercept_ = 1.5

        X = np.array(X)
        y = np.array(y)

        for _ in range(self.n_epochs):
            linear_preds = X @ self.coef_ + self.intercept_
            preds = self.sigmoid(linear_preds)

            grad_w0 = 1/len(X) * np.sum(preds - y)
            grad_w = 1/len(X) * X.T@(preds-y)

            self.coef_ = self.coef_ - self.learning_rate * grad_w
            self.intercept_ =  self.intercept_ - self.learning_rate * grad_w0

    def predict(self, X):
        linear_preds = X@self.coef_ + self.intercept_
        preds = self.sigmoid(linear_preds)
        return np.array([0 if pred <= 0.5 else 1 for pred in preds])
    
    def predict_propa(self, X):
        linear_preds = X@self.coef_ + self.intercept_
        preds = self.sigmoid(linear_preds)
        return round(preds, 2)
    
    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    



st.title('Данный веб-сервис позволяет построить логистическую регрессию с любым количеством фичей, начиная от двух')

st.header('Пример использования на предподготовленных данных')


train = pd.read_csv('data/credit_train.csv')
test = pd.read_csv('data/credit_test.csv')


ss = StandardScaler()


x_train, y_train, x_test, y_test = train.drop('Personal.Loan', axis = 1), train['Personal.Loan'], test.drop('Personal.Loan', axis = 1), test['Personal.Loan']

x_train = pd.DataFrame(ss.fit_transform(x_train), columns=x_train.columns)

x_test = pd.DataFrame(ss.transform(x_test), columns=x_test.columns)


n_epochs = st.slider('Выберите кол-во эпох:', 1, 2000)

my_model = LogReg(0.01, n_epochs)

my_model.fit(x_train, y_train)

coefs, svob = my_model.coef_, my_model.intercept_

preds = my_model.predict(x_test)

accuracy = my_model.score(y_test, preds)


st.subheader(f'Результат обучения собственной модели на {n_epochs} эпохах:')


st.markdown(f'### свободный член: **{round(svob, 3)}**\n ### остальные коэфы: **w1: {round(coefs[0], 3)}**, **w2: {round(coefs[1], 3)}**')

col1, col2 = st.columns(2)

col2.metric(label = 'Our accuracy', value = round(accuracy, 3))

fig, ax = plt.subplots()
ax = plot_decision_regions(X=np.array(x_train), y = np.array(y_train), clf = my_model, legend = 2)


col1.pyplot(fig)


st.title('Давайте теперь попробуем на ваших данных!')

data = st.file_uploader("Upload a CSV")

if data:
    df = pd.read_csv(data)

    st.subheader('Введите название целевой переменной')

    target = st.text_input('target label')
    if target:

        scaler = StandardScaler()

        X = df.drop(target, axis=1)
        y = df[target]

        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        st.subheader('Выберите следующие параметры модели:')
        col1, col2, col3 = st.columns(3)
        test_size = col1.slider('Test size', 0.1, 0.9)
        lr = col2.slider('Lerning rate', 0.0001, 0.1)
        n_epochs = col3.slider('Epochs', 1, 2000)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        logreg = LogReg(lr, n_epochs)

        logreg.fit(X_train, y_train)


        preds = logreg.predict(X_test)
        accuracy = logreg.score(y_test, preds)

        if len(X_test.columns) == 2:
            st.markdown('### Результат работы модели')
            col4, col5 = st.columns(2)


            col5.metric(label = ' Accuracy на тестовой выборке', value = round(accuracy, 3))
            if accuracy > 0.8:
                st.subheader('Весьма неплохо!')
            
            else:
                st.subheader('Могло быть и лучше...')

            fig, ax = plt.subplots()
            ax = plot_decision_regions(X=np.array(X_train), y = np.array(y_train), clf = logreg, legend = 2)


            col4.pyplot(fig)

        else:
            st.markdown('### Результат работы модели')
            st.metric(label = 'Accuracy на тестовой выборке', value = round(accuracy, 3))
            if accuracy > 0.8:
                st.subheader('Весьма неплохо!')
            
            else:
                st.subheader('Могло быть и лучше...')
            
            




