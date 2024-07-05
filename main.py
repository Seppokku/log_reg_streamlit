
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

class LogReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.normal(1, 0.5, size=n_inputs)
        self.intercept_ = 1.5

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, X, y, n_epochs):

        X = np.array(X)
        y = np.array(y)

        for _ in range(n_epochs):
            linear_preds = X @ self.coef_ + self.intercept_
            preds = self.sigmoid(linear_preds)

            grad_w0 = 1/len(X) * np.sum(preds - y)
            grad_w = 1/len(X) * X.T@(preds-y)

            self.coef_ = self.coef_ - self.learning_rate * grad_w
            self.intercept_ =  self.intercept_ - self.learning_rate * grad_w0

    def predict(self, X):
        linear_preds = X@self.coef_ + self.intercept_
        preds = self.sigmoid(linear_preds)
        return [0 if pred <= 0.5 else 1 for pred in preds]
    
    def predict_propa(self, X):
        linear_preds = X@self.coef_ + self.intercept_
        preds = self.sigmoid(linear_preds)
        return round(preds, 2)
    
    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    



st.title('Сравнение Home made модели логистической регрессии и модели из пакета scikit-learn')

st.subheader('Mы обучались на подготовленных данных и нам нужно было предсказать шанс возврата кредита банку клиентом')
st.subheader('Вот что у нас получилось:')


train = pd.read_csv('data/credit_train.csv')
test = pd.read_csv('data/credit_test.csv')


ss = StandardScaler()


x_train, y_train, x_test, y_test = train.drop('Personal.Loan', axis = 1), train['Personal.Loan'], test.drop('Personal.Loan', axis = 1), test['Personal.Loan']

x_train = pd.DataFrame(ss.fit_transform(x_train), columns=x_train.columns)

x_test = pd.DataFrame(ss.transform(x_test), columns=x_test.columns)


n_epochs = 2000

my_model = LogReg(0.01, 2)

my_model.fit(x_train, y_train, n_epochs)

coefs, svob = my_model.coef_, my_model.intercept_

preds = my_model.predict(x_test)

accuracy = my_model.score(y_test, preds)


st.header(f'Результат обучения собственной модели на {n_epochs} эпохах')

st.subheader('Веса нашей модели:')

st.markdown(f'свободный член: **{svob}**\n остальные коэфы: **w1: {coefs[0]}**, **w2: {coefs[1]}**')
st.title('Сравнение нашей accuracy нашей модели и модели из коробки:')

lr = LogisticRegression()
lr.fit(x_train, y_train)
coefs, svob = lr.coef_, lr.intercept_

preds = lr.predict(x_test)

accuracy_2 = accuracy_score(y_test, preds)
col_1, col_2 = st.columns(2)
col_1.metric(label = 'Our accuracy', value = round(accuracy, 3))
col_2.metric('scikit LogReg accuracy', value = round(accuracy_2, 3), delta=f'{round((round(accuracy_2, 3)-round(accuracy, 3))/round(accuracy, 3)*100, 2)}%')


st.subheader('Веса модели из коробки:')
st.markdown(f'свободный член: **{svob[0]}**\n остальные коэфы: **w1: {coefs[0][0]}**, **w2: {coefs[0][1]}**')






