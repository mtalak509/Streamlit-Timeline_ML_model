import pandas as pd
import numpy as np
import streamlit as st

import sklearn
sklearn.set_config(transform_output="pandas")

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

plt.style.use('seaborn-v0_8')

# Заголовок приложения
st.title('Анализ данных и предсказание временного ряда')

# Создание боковой панели
st.sidebar.header('Загрузка данных')

# Компонент загрузки файла
uploaded_file = st.sidebar.file_uploader(
    'Загрузите xls файл',
    type=['xls'],
    help='Поддерживаются форматы xls'
)

# Отображение информации о загрузке
if uploaded_file is not None:
    # Определение типа файла
    file_extension = uploaded_file.name.split('.')[-1]

    try:
        # Загрузка данных
        if file_extension == 'xls':
            df = pd.read_excel(uploaded_file)

        # Uploading and cleaning
        # df = pd.read_excel('/home/maxim/DS/Phase_1/ds-phase-1/06-unsupervised/aux/Sample - Superstore.xls')
        df = df[df['Category'] == 'Furniture']

        data = df[['Order Date', 'Sales']]
        data = data.sort_values(by='Order Date', ascending=True)
        data = data.set_index('Order Date')

        weekly_sum = data.resample('W').sum()

        st.subheader('Предпросмотр данных') #Data show
        st.dataframe(weekly_sum.head())

        st.subheader('График продаж по неделям') #Initial plot

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=weekly_sum, x='Order Date', y='Sales', ax=ax)
        st.pyplot(fig)

        #Decomposing

        decomposition_add = seasonal_decompose(weekly_sum['Sales'], 
                                            model='additive', 
                                            period=52)

        st.subheader('Декомпозиция временного ряда')
        fig = decomposition_add.plot()
        fig.set_size_inches(12, 8)
        # plt.suptitle('Аддитивная декомпозиция временного ряда', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

        #Autocorrelation
        st.subheader('Автокорреляция (lags=52)')
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(weekly_sum['Sales'], ax=axes[0], lags=52)
        plot_pacf(weekly_sum['Sales'], ax=axes[1], lags=52)
        plt.legend()
        st.pyplot(fig)

        #Prophet model
        data_prophet = weekly_sum['Sales'].reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'}) # Обязательное имена колнок с датой и таргетотм в prophet

        data_train = data_prophet[data_prophet['ds'].dt.year < 2017]
        data_test = data_prophet[data_prophet['ds'].dt.year >= 2017]

        model = Prophet()
        model.fit(data_train)

        seasonality_period = 52
        number_of_future_predicted_points = 2 * seasonality_period

        future = model.make_future_dataframe(periods=number_of_future_predicted_points, freq='W')
        forecast = model.predict(future)

        #Prophet plot
        st.subheader('Прогнозирование продаж')
        forecast_train = forecast[:-number_of_future_predicted_points] # Трейновый период
        forecast_test = forecast[-number_of_future_predicted_points: -number_of_future_predicted_points + len(data_test)] # Тестовый
        forecast_future = forecast[-number_of_future_predicted_points + len(data_test):] # Будущий период

        prophet_mae_train = np.round(mean_absolute_error(data_train['y'], forecast_train['yhat']), 1)
        prophet_mae_test = np.round(mean_absolute_error(data_test['y'], forecast_test['yhat']), 1)
        prophet_r2_train = np.round(r2_score(data_train['y'], forecast_train['yhat']), 1)


        st.markdown("---")

        # Метрики в колонках
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE на трейне", f"{prophet_mae_train:.2f}")
        with col2:
            st.metric("MAE на тесте", f"{prophet_mae_test:.2f}")
        with col3:
            st.metric("R2", f"{prophet_r2_train:.2f}")
        with col4:
            st.metric("Модель", "Prophet")

        # Создаем график
        fig, ax = plt.subplots(figsize=(14, 8))

        # Построение графиков
        ax.plot(weekly_sum['Sales'], label='Реальные данные', linewidth=2.5, color='#2E86AB')
        ax.plot(forecast_train['ds'], forecast_train['yhat'], marker='v', linestyle=':', 
                linewidth=2, label=f'Прогноз на трейне', color='#A23B72')
        ax.plot(forecast_test['ds'], forecast_test['yhat'], marker='v', linestyle=':', 
                linewidth=2, label=f'Прогноз на тесте', color='#F18F01')
        ax.plot(forecast_future['ds'], forecast_future['yhat'], marker='v', linestyle=':', 
                linewidth=2.5, label='Будущий прогноз', color='#C73E1D')

        # Область доверительного интервала
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='gray', alpha=0.2, label='95% доверительный интервал')

        # Форматирование
        ax.set_title(f'Прогноз продаж - Модель Prophet\nСредняя абсолютная ошибка на тесте: {prophet_mae_test:.2f}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Период', fontsize=12, fontweight='bold')
        ax.set_ylabel('Объем продаж', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Отображение графика
        st.pyplot(fig)


    except Exception as e:
        st.error(f'Произошла ошибка при загрузке файла: {str(e)}')
        
else:
    st.warning('Пожалуйста, загрузите файл для продолжения работы')