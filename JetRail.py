import pandas as pd
from fbprophet import Prophet

# 加载数据
train = pd.read_csv('./train.csv')
print(train.head())
    
# 转换为pandas中的日期格式
train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train.Datetime
print(train.head())
    
# 去掉ID,Datetime字段
train.drop(['ID', 'Datetime'], axis=1, inplace=True)
print(train.head())
    
# 按照天 进行采样
daily_train = train.resample('D').sum()
print(daily_train.head())
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Count
daily_train.drop(['Count'], axis=1, inplace=True)
print(daily_train.head())
    

# 拟合prophet模型
m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
m.fit(daily_train)
# 预测未来7个月213天
future = m.make_future_dataframe(periods=213)
forecast = m.predict(future)
print(forecast)


    # 查看
    m.plot(forecast)
    # 查看成分
    m.plot_components(forecast)
