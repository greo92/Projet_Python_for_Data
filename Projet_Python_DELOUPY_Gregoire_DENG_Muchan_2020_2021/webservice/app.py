from flask import Flask, render_template, request
import pickle
import pandas as pd


model = pickle.load(open('bike.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('homepage.html')

@app.route('/predict',methods=['POST'])
def home():

    df = pd.read_csv('SeoulBikeData.csv', sep=',')
    df = df.drop(['Rented Bike Count'], axis=1)
    columns = ('Date',
               'Hour',
               'Temperature(?C)',
               'Humidity(%)',
               'Wind speed (m/s)',
               'Visibility (10m)',
               'Dew point temperature(?C)',
               'Solar Radiation (MJ/m2)',
               'Rainfall(mm)',
               'Snowfall (cm)',
               'Seasons',
               'Holiday',
               'Functioning Day')


    Date = request.form['date']
    Hour = int(request.form['hour'])
    Temperature = float(request.form['temperature'])
    Humidity = int(request.form['humidity'])
    Windspeed = float(request.form['windspeed'])
    Visibility = int(request.form['visibility'])
    Dewpointtemperature = float(request.form['dewpointtemperature'])
    SolarRadiation = float(request.form['solarradiation'])
    Rainfall = float(request.form['rainfall'])
    Snowfall = float(request.form['snowfall'])
    Seasons = request.form['seasons']
    Holiday = request.form['holiday']
    FunctioningDay = request.form['functioningday']

    data = [[Date, Hour, Temperature, Humidity, Windspeed, Visibility, Dewpointtemperature,
             SolarRadiation, Rainfall, Snowfall, Seasons, Holiday, FunctioningDay]]

    copy_df = df.loc[0:0,columns].copy()
    copy_df.loc[0, columns] = Date, Hour, Temperature, Humidity, Windspeed, Visibility, Dewpointtemperature,SolarRadiation, Rainfall, Snowfall, Seasons, Holiday, FunctioningDay
    new_df = copy_df.append(df, sort=False).reset_index(drop=True)
    df = new_df

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
    df = df.drop(['Date'], axis=1)
    df = pd.get_dummies(df, columns=['month'], drop_first=True, prefix='month')
    df = pd.get_dummies(df, columns=['Seasons'], drop_first=True, prefix='season')
    df = pd.get_dummies(df, columns=['Holiday'], drop_first=True, prefix='holiday')
    df = pd.get_dummies(df, columns=['Functioning Day'], drop_first=True, prefix='fctday')
    df = pd.get_dummies(df, columns=['weekday'], drop_first=True, prefix='wday')
    df = pd.get_dummies(df, columns=['quarter'], drop_first=True, prefix='qrtr')
    df = pd.get_dummies(df, columns=['is_month_start'], drop_first=True, prefix='m_start')
    df = pd.get_dummies(df, columns=['is_month_end'], drop_first=True, prefix='m_end')
    df = df.head(1)


    pred = int(model.predict(df))

    return render_template('after.html',data = pred)

if __name__ == '__main__':
    app.run(debug=True)
