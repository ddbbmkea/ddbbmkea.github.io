## 3. Collaborative Filtering (협업 필터링 : 사용자 리뷰 기반)

나 : Up, 주토피아 재밌게 봄\
다른 사람 : Up, 주토피아, 인사이드 아웃 재밌게 봄
    
협업 필터링은 나에게 인사이드 아웃 추천


```python
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
```


```python
ratings = pd.read_csv('ratings_small.csv')
ratings.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>




```python
# 최저 점수
ratings['rating'].min()
```




    0.5




```python
# 최고 점수
ratings['rating'].max()
```




    5.0




```python
# rating scale을 0.5~5점 사이로 변겅
reader = Reader(rating_scale=(0.5, 5))
```


```python
# userid, itemid, rating 3가지 컬럼만 가져야 함 (라이브러리에서 그렇게 제안하기 때문)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader) # reader는 앞에 만든 reader 패키지로 입력
# timestamp 컬럼까지 넣으면 에러 발생
data
```




    <surprise.dataset.DatasetAutoFolds at 0x2e0fd0ef6a0>




```python
svd = SVD(random_state=0)
```


```python
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True) # verbose=True일 경우 진행 과정 보여 줌
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.9011  0.8991  0.9042  0.8836  0.9031  0.8982  0.0075  
    MAE (testset)     0.6927  0.6921  0.6958  0.6811  0.6945  0.6912  0.0052  
    Fit time          3.40    3.35    3.34    3.33    3.37    3.36    0.02    
    Test time         0.10    0.10    0.10    0.15    0.15    0.12    0.02    
    




    {'test_rmse': array([0.90111218, 0.89912163, 0.90415043, 0.88359237, 0.90307726]),
     'test_mae': array([0.69265447, 0.69207625, 0.69575195, 0.68106757, 0.69450529]),
     'fit_time': (3.401918411254883,
      3.350374698638916,
      3.339364767074585,
      3.3343608379364014,
      3.371392250061035),
     'test_time': (0.10458993911743164,
      0.10258698463439941,
      0.10258841514587402,
      0.14762639999389648,
      0.15363168716430664)}



RMSE : 0.90\
MAE : 0.69\
학습 시간 : 3.4초\
테스트 시간 : 0.1초

교차 검증 (K-Fold 교차 검증)

100개 데이터

5개의 세트로 나뉨\
A : 1 - 20 번째 데이터\
B : 21 - 40 번째 데이터\
C : 41 - 60 번째 데이터\
D : 61 - 80 번째 데이터\
E : 81 - 100 번째 데이터

ABCD (train set) E (test set)\
ABCE (train set) D (test set)\
ABDE (train set) C (test set)\
ACDE (train set) B (test set)\
BCDE (train set) A (test set)\
-> 4개 세트 훈련, 1개 세트 테스트

5번 테스트 하고 나서\
모든 결과의 평균으로 성능을 평가하는 것이 교차 검증


```python
trainset = data.build_full_trainset()
svd.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x2e0fd094df0>




```python
# userId = 1인 사람의 데이터
ratings[ratings['userId'] == 1]
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1263</td>
      <td>2.0</td>
      <td>1260759151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1287</td>
      <td>2.0</td>
      <td>1260759187</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1293</td>
      <td>2.0</td>
      <td>1260759148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1339</td>
      <td>3.5</td>
      <td>1260759125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1343</td>
      <td>2.0</td>
      <td>1260759131</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1371</td>
      <td>2.5</td>
      <td>1260759135</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1405</td>
      <td>1.0</td>
      <td>1260759203</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1953</td>
      <td>4.0</td>
      <td>1260759191</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2105</td>
      <td>4.0</td>
      <td>1260759139</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2150</td>
      <td>3.0</td>
      <td>1260759194</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2193</td>
      <td>2.0</td>
      <td>1260759198</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2294</td>
      <td>2.0</td>
      <td>1260759108</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2455</td>
      <td>2.5</td>
      <td>1260759113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2968</td>
      <td>1.0</td>
      <td>1260759200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3671</td>
      <td>3.0</td>
      <td>1260759117</td>
    </tr>
  </tbody>
</table>




```python
# userId = 1이 movieId = 302에 몇 점을 줄 것인가
svd.predict(1, 302)
# 평가할 것으로 예측되는 점수는 2.7점
```




    Prediction(uid=1, iid=302, r_ui=None, est=2.7142061734434044, details={'was_impossible': False})




```python
# 실제로 userId = 1이 movieId = 302에 3점을 부여했지만, 모델이 예측한 점수는 2.7점
svd.predict(1, 302, 3) # 3은 굳이 입력하지 않아도 됨
```




    Prediction(uid=1, iid=302, r_ui=3, est=2.7142061734434044, details={'was_impossible': False})




```python
# userId = 1이 movieId = 1029에 대해 실제 평가 3점일 때, 모델이 예측한 평가 점수는 2.88점 (상당히 근접)
svd.predict(1, 1029, 3)
```




    Prediction(uid=1, iid=1029, r_ui=3, est=2.8814455446761933, details={'was_impossible': False})




```python
# userId = 100인 사람의 데이터
ratings[ratings['userId'] == 100]
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15273</th>
      <td>100</td>
      <td>1</td>
      <td>4.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15274</th>
      <td>100</td>
      <td>3</td>
      <td>4.0</td>
      <td>854194024</td>
    </tr>
    <tr>
      <th>15275</th>
      <td>100</td>
      <td>6</td>
      <td>3.0</td>
      <td>854194023</td>
    </tr>
    <tr>
      <th>15276</th>
      <td>100</td>
      <td>7</td>
      <td>3.0</td>
      <td>854194024</td>
    </tr>
    <tr>
      <th>15277</th>
      <td>100</td>
      <td>25</td>
      <td>4.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15278</th>
      <td>100</td>
      <td>32</td>
      <td>5.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15279</th>
      <td>100</td>
      <td>52</td>
      <td>3.0</td>
      <td>854194056</td>
    </tr>
    <tr>
      <th>15280</th>
      <td>100</td>
      <td>62</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15281</th>
      <td>100</td>
      <td>86</td>
      <td>3.0</td>
      <td>854194208</td>
    </tr>
    <tr>
      <th>15282</th>
      <td>100</td>
      <td>88</td>
      <td>2.0</td>
      <td>854194208</td>
    </tr>
    <tr>
      <th>15283</th>
      <td>100</td>
      <td>95</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15284</th>
      <td>100</td>
      <td>135</td>
      <td>3.0</td>
      <td>854194086</td>
    </tr>
    <tr>
      <th>15285</th>
      <td>100</td>
      <td>141</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15286</th>
      <td>100</td>
      <td>608</td>
      <td>4.0</td>
      <td>854194024</td>
    </tr>
    <tr>
      <th>15287</th>
      <td>100</td>
      <td>648</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15288</th>
      <td>100</td>
      <td>661</td>
      <td>3.0</td>
      <td>854194086</td>
    </tr>
    <tr>
      <th>15289</th>
      <td>100</td>
      <td>708</td>
      <td>3.0</td>
      <td>854194056</td>
    </tr>
    <tr>
      <th>15290</th>
      <td>100</td>
      <td>733</td>
      <td>3.0</td>
      <td>854194024</td>
    </tr>
    <tr>
      <th>15291</th>
      <td>100</td>
      <td>736</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15292</th>
      <td>100</td>
      <td>745</td>
      <td>4.0</td>
      <td>854194208</td>
    </tr>
    <tr>
      <th>15293</th>
      <td>100</td>
      <td>780</td>
      <td>3.0</td>
      <td>854193977</td>
    </tr>
    <tr>
      <th>15294</th>
      <td>100</td>
      <td>786</td>
      <td>3.0</td>
      <td>854194056</td>
    </tr>
    <tr>
      <th>15295</th>
      <td>100</td>
      <td>802</td>
      <td>4.0</td>
      <td>854194111</td>
    </tr>
    <tr>
      <th>15296</th>
      <td>100</td>
      <td>1073</td>
      <td>5.0</td>
      <td>854194056</td>
    </tr>
    <tr>
      <th>15297</th>
      <td>100</td>
      <td>1356</td>
      <td>4.0</td>
      <td>854194086</td>
    </tr>
  </tbody>
</table>





```python
# userId = 100이 movieId = 1029에 대해 3.77점 주리라 예측
# userId = 1에 비해 높음
svd.predict(100, 1029)
```




    Prediction(uid=100, iid=1029, r_ui=None, est=3.7705476478414846, details={'was_impossible': False})

