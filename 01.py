import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
          '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
          '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '21월', '22요일', '23주말 및 공휴일',
          '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
          '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
          '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)', '41Year']
# df.columns = header_english
# header_english = [
#     '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
#     '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
#     '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
#     '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
#     '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
#     '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
#     '28Night_Clear Percentage', 'Night_Cloudy Percentage', 'Night_Rain Percentage', 'Night_Lightning Presence',
#     'Difference in High Temp from Previous Day', 'Difference in Average Temp from Previous Day', 'Difference in Low Temp from Previous Day',
#     '5-Day Moving Average of High Temp', '5-Day Moving Average of Average Temp', '5-Day Moving Average of Apparent Temp',
#     '5-Day Moving Average of Discomfort Index', 'Transported Patients on Previous Day', '5-Day Moving Average of Transported Patients', 'Year'
# ]
df.columns = header

# 3. 변경된 CSV 파일 저장
df.to_csv('./data/6.Heatstroke_train_new_header.csv', index=False)

# 저장된 CSV 파일 다시 읽기
data = pd.read_csv('./data/6.Heatstroke_train_new_header.csv')


# '이송 인원' 열을 기준으로 정렬
sorted_df_by_transport = df.sort_values(by='1날짜 및 시간')

# 정렬된 데이터 출력 (열의 간격이 밀리지 않도록)
with pd.option_context('display.max_columns', None):  # 모든 열을 출력
    print("Sorted by '1날짜 및 시간':")
    print(sorted_df_by_transport.head())

#__________________________21.월_________________________________________________________
#
# # X와 Y 설정
# X5 = data['21월']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Discomfort Index vs Number of Heatstroke Patients")
# plt.xlabel("Discomfort Index")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()