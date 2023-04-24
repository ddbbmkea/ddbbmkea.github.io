# 2. Multiple Linear Regression

y = b + m1x1 + m2x2 + ... + mnxn

### 원-핫 인코딩


```python
import pandas as pd

dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X, y
```




    (array([[0.5, 3, 'Home'],
            [1.2, 4, 'Library'],
            [1.8, 2, 'Cafe'],
            [2.4, 0, 'Cafe'],
            [2.6, 2, 'Home'],
            [3.2, 0, 'Home'],
            [3.9, 0, 'Library'],
            [4.4, 0, 'Library'],
            [4.5, 5, 'Home'],
            [5.0, 1, 'Cafe'],
            [5.3, 2, 'Cafe'],
            [5.8, 0, 'Cafe'],
            [6.0, 3, 'Library'],
            [6.1, 1, 'Cafe'],
            [6.2, 1, 'Library'],
            [6.9, 4, 'Home'],
            [7.2, 2, 'Cafe'],
            [8.4, 1, 'Home'],
            [8.6, 1, 'Library'],
            [10.0, 0, 'Library']], dtype=object),
     array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
             68,  72,  58,  76,  86,  90, 100], dtype=int64))




```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transformers 전달값에 리스트 형태로 값 넣어줌
# 튜플 형태로 값 3개 넣음
# 첫 번째 값: 우리가 어떤 변경을 가할지 (encoder)
# 두 번째 값: 인코딩을 수행할 클래스 객체 (OneHotEncoder)
# 다중공선성 문제를 없애기 위해 drop='first' => 첫 번째 칼럼은 drop 돼서 n개 피쳐 중 n-1개만 사용
# Home, Library, Cafe 중 Cafe 제거
#        Home Library Cafe
# Home : 1 0 0 -> 1 0 수정
# Library : 0 1 0 -> 0 1 수정
# Cafe : 0 0 1 -> 0 0 수정
# 세 번째 값: 어떤 값으로 원핫인코딩을 할지 칼럼 인덱스 지정 (0, 1, 2 중 2에 해당하는 장소 피쳐)
# remainder: 원핫인코딩을 하지 않는 나머지 칼럼들은 어떻게 하는지 (passthrough: 그냥 둔다)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)

X
```




    array([[1.0, 0.0, 0.5, 3],
           [0.0, 1.0, 1.2, 4],
           [0.0, 0.0, 1.8, 2],
           [0.0, 0.0, 2.4, 0],
           [1.0, 0.0, 2.6, 2],
           [1.0, 0.0, 3.2, 0],
           [0.0, 1.0, 3.9, 0],
           [0.0, 1.0, 4.4, 0],
           [1.0, 0.0, 4.5, 5],
           [0.0, 0.0, 5.0, 1],
           [0.0, 0.0, 5.3, 2],
           [0.0, 0.0, 5.8, 0],
           [0.0, 1.0, 6.0, 3],
           [0.0, 0.0, 6.1, 1],
           [0.0, 1.0, 6.2, 1],
           [1.0, 0.0, 6.9, 4],
           [0.0, 0.0, 7.2, 2],
           [1.0, 0.0, 8.4, 1],
           [0.0, 1.0, 8.6, 1],
           [0.0, 1.0, 10.0, 0]], dtype=object)



앞 2개는 장소 나타냄\
뒤 2개는 원래 데이터 앞 칼럼 2개 (hour, absent)

1 0 : Home\
0 1 : Library\
0 0 : Cafe

### 데이터 세트 분리


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### 학습 (다중 선형 회귀)


```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```







### 예측값과 실제값 비교 (테스트 세트)


```python
# 예측값
y_pred = reg.predict(X_test)
y_pred
```




    array([ 92.15457859,  10.23753043, 108.36245302,  38.14675204])




```python
# 실제값
y_test
```




    array([ 90,   8, 100,  38], dtype=int64)




```python
# 집, 도서관, 공부시간, 결석횟수에 따른 계수
reg.coef_
```




    array([-5.82712824, -1.04450647, 10.40419528, -1.64200104])



집에서 공부하면 -5.8\
도서관에서 공부하면 -1.0\
카페에서 공부하면 0\
=> 카페에서 공부했을 때 성적에 가장 도움이 됨

공부시간 1시간 증가할 때마다 10점씩 상승\
결석 1번 할 때마다 1.6씩 성적에 안 좋은 영향


```python
reg.intercept_
```




    5.365006706544811



### 모델 평가


```python
reg.score(X_train, y_train) # 훈련 세트
```




    0.9623352565265527




```python
reg.score(X_test, y_test) # 테스트 세트
```




    0.9859956178877447



### 다양한 평가 지표 (회귀 모델)

1. MAE (Mean Absolute Error) : 실제값과 예측값 차이의 절대값
1. MSE (Mean Squared Error) : 실제값과 예측값 차이의 제곱
1. RMSE (Root Mean Squared Error) : 실제값과 예측값 차이의 제곱에 루트
1. R2 : 결정 계수

> R2는 1에 가까울수록, 나머지는 0에 가까울수록 좋음


```python
from sklearn.metrics import mean_absolute_error

# MAE
mean_absolute_error(y_test, y_pred) # 실제값, 예측값
```




    3.225328518828783




```python
from sklearn.metrics import mean_squared_error

# MSE
mean_squared_error(y_test, y_pred)
```




    19.90022698151482




```python
# RMSE
mean_squared_error(y_test, y_pred, squared=False)
```




    4.460967045553556




```python
from sklearn.metrics import r2_score

# R2
r2_score(y_test, y_pred)
```




    0.9859956178877447



Linear Regression의 score는 R2로 계산됨
