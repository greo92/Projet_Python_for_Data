import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('SeoulBikeData.csv', sep=',')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day
df['dayofyear'] = pd.DatetimeIndex(df['Date']).dayofyear
df['weekofyear'] = pd.DatetimeIndex(df['Date']).weekofyear
df['weekday'] = pd.DatetimeIndex(df['Date']).weekday
df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
df['is_month_start'] = pd.DatetimeIndex(df['Date']).is_month_start
df['is_month_end'] = pd.DatetimeIndex(df['Date']).is_month_end
df = df.drop(['Date'], axis = 1)


df = pd.get_dummies(df, columns=['month'], drop_first=True, prefix='month')
df = pd.get_dummies(df, columns=['Seasons'], drop_first=True, prefix='season')
df = pd.get_dummies(df, columns=['Holiday'], drop_first=True, prefix='holiday')
df = pd.get_dummies(df, columns=['Functioning Day'], drop_first=True, prefix='fctday')
df = pd.get_dummies(df, columns=['weekday'], drop_first=True, prefix='wday')
df = pd.get_dummies(df, columns=['quarter'], drop_first=True, prefix='qrtr')

df = pd.get_dummies(df, columns=['is_month_start'], drop_first=True, prefix='m_start')

df = pd.get_dummies(df, columns=['is_month_end'], drop_first=True, prefix='m_end')


X = df.drop(['Rented Bike Count'], axis = 1) 
Y = df['Rented Bike Count']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=1)

ss_x = preprocessing.StandardScaler()
X_train_st = ss_x.fit_transform(X_train)
X_test_st = ss_x.transform(X_test)

ss_y = preprocessing.StandardScaler()
Y_train_st = ss_y.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_st = ss_y.transform(Y_test.values.reshape(-1, 1))


regressor = GradientBoostingRegressor(learning_rate=0.2,
                                      max_depth=8,
                                      max_features=0.8,
                                      min_samples_leaf=9,
                                      n_estimators=80).fit(X_train, Y_train.ravel())


pickle.dump(regressor, open('bike.pkl', 'wb'))





















