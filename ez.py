import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

file_path = 'E:\\staj\\gotovoe\\sales-data-sample.csv'
data = pd.read_csv(file_path)

data['OrderDate'] = pd.to_datetime(data['OrderDate'])


data.rename(columns={'OrderDate': 'ds'}, inplace=True)


data['ds'] = data['ds'].dt.tz_localize(None)


data['OriginalOrder'] = data.index

variables = ['Sales', 'SalesForecast', 'Profit', 'Quantity']
daily_data = {}

for var in variables:
    var_data = data.groupby(['ds', 'Category']).agg({var: 'sum'}).reset_index()
    var_data.rename(columns={var: 'y'}, inplace=True)


    var_data['ds'] = var_data['ds'].dt.tz_localize(None)


    var_data = var_data.merge(data[['ds', 'Category', 'OriginalOrder']], on=['ds', 'Category'], how='left')

    daily_data[var] = var_data

models = {}
forecasts = {}
results = {}

periods = {'3_months': 90, '6_months': 180, '1_year': 365}

for var, var_data in daily_data.items():

    model = Prophet()
    model.fit(var_data)
    models[var] = model

    for period_name, horizon in periods.items():
        try:
            df_cv = cross_validation(model, initial='365 days', period='90 days', horizon=f'{horizon} days')
            df_p = performance_metrics(df_cv)
            df_cv.to_csv(f'crossvalidation_{var}_{period_name}.csv', index=False)
            df_p.to_csv(f'performancemetrics_{var}_{period_name}.csv', index=False)
        except Exception as e:
            print(f"Ошибка при кросс-валидации для {var} и периода {period_name}: {e}")

    future = model.make_future_dataframe(periods=365)


    future['ds'] = future['ds'].dt.tz_localize(None)

    forecast = model.predict(future)


    forecast = forecast.merge(var_data[['ds', 'Category', 'OriginalOrder']], on='ds', how='left')


    forecast = forecast.sort_values(by='OriginalOrder')

    forecast.to_csv(f'forecast_{var}.csv', index=False)
