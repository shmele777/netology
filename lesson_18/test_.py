import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#https://proglib.io/p/linear-regression
#https://habr.com/ru/post/279117/
#https://coderoad.ru/52961074/%D0%9A%D0%B0%D0%BA-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B8%D1%82%D1%8C-%D0%B4%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8E-%D0%BE%D1%81%D1%82%D0%B0%D1%82%D0%BA%D0%BE%D0%B2-%D0%BF%D0%BE%D1%81%D0%BB%D0%B5-%D0%BF%D0%BE%D0%B4%D0%B3%D0%BE%D0%BD%D0%BA%D0%B8-%D0%BB%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%BE%D0%B9-%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D0%B8-%D1%81-%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E

water_df = pd.read_csv('lesson_18/water.csv')

water_df.plot(kind='scatter', x='mortality', y='hardness')

# Построим точечный график
plt.show()

# Расчитаем коэффициент корреляции Пирсона и Спирмана
pirson_df = water_df[['mortality', 'hardness']].corr()
spearman_df = water_df[['mortality', 'hardness']].corr(method='spearman')
kendall_df = water_df[['mortality', 'hardness']].corr(method='kendall')

print('коэффициенты корреляции Пирсона:', pirson_df['hardness']['mortality'])
print('коэффициенты корреляции Спирмана:', spearman_df['hardness']['mortality'])
print('коэффициенты корреляции Кендалла:', spearman_df['hardness']['mortality'])




x = water_df[['mortality']]
y = water_df['hardness']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

print(x_train.shape)
print(y_train.shape)

model = LinearRegression()
result = model.fit(x_train, y_train)
print(result.summary())

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x_test)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, c='r')
plt.show()

print('Коэффициент детерминации:', model.score(x_test, y_test))





