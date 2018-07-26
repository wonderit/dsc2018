# DSC2018 - CDAL

## 1. Online Test

### Problem Edited : 

- Problem link : https://www.kaggle.com/annavictoria/speed-dating-experiment/home

- Dataset : https://www.kaggle.com/annavictoria/speed-dating-experiment#Speed%20Dating%20Data.csv

- train : wave 1 ~ 15

- test : wave 16 ~ 21 

- Baseline : https://www.kaggle.com/jph84562/the-ugly-truth-of-people-decisions-in-speed-dating

### Story line 
1. Introduction

    - Speed date dataset 선정 동기 (원석)

    - dataset 설명 (동현)
        - 2002~2004년동안 21 차례에 거친 speed dating 실험을 한 결과. 각각의 실험자는 상대 성별의 모든 상대들과 데이트를 4분간 진행한다. 
            해당 dataset은 데이트 이전과 이후의 설문 문항에 대한 대답으로 이루어져있다.

    - dataset 분석 (동현)
    
    - 문제 제기 : 
        
          ( 나에 대한 평가 없이 ) 상대방이 나를 마음에 들어하는지 여부를 예측

2. 가설 제시 (유중)

    - 상대방이 나를 마음에 들어했던 date 중 상대방이 비중을 두는 요소 
    (attractive, sincere, intelligence, funny, ambitious, shared interests ) 와 
    내가 나를 평가한 점수에 대한 관계

    - 내가 나를 평가한 점수(요소)와 상대가 나를 평가한 점수(요소_partner)에 대한 관계

    - 상대가 나를 평가한 점수(요소_partner)와 상대가 나를 마음에 들어했는지(decision_o)에 대한 관계

    + a

3. 해결 방안

    - dataset preprocessing (원석)
        - 사용한 변수
        - null 데이터 처리 (train/test : 6428 / 1950 -> 5028 / 1572)
        - train/test 분리

    - 학습 모델 (다같이)
        - blending, ensemble 
        - Logistic, Randomforest, ANN classifier 
        - crossvalidation 
        
    - 학습 목표 

            speed date 이후 상대방에 대한 나의 평가를 이용해 상대방이 나를 마음에 들어하는지 여부 예측

4. 결과 (다같이)

    - accuracy 비교 
        * wonsuk
            
            Mean Accuracy Result (w/o opponent's score data) for decision_o
            ----------------------------
            |Model          | w/o Scaler| StandardScaler    | MinMaxScaler      | QuantileTransformer   | PowerTransformer  |
            |:----:         | :----:    | :----:            | :----:            | :----:                |:----:
            | KNN(k=8)      | 54.4      | **58.9**          | 57.1              | 56.2                  | 57.2
            | LR            | 59.7      | 59.3              | 59.5              | 58.5                  | **60.1**
            | Linear SVM    | 43.7      | **60.1**          | **60.1**          | 58.8                  | 59.9
            | RBF SVM       | 56.8      | 58.4              | **59.4**          | **59.4**              | 57.4
            | DecisionTree  | **58.0**  | 55.9              | 56.0              | 55.2                  | 57.1 
            | RF            | **58.6**  | 57.6              | 57.8              | 57.6                  | 56.8
            | MLP(5,2)      | 57.3      | 60.6              | **59.3**          | **59.3**              | 58.7
            | AdaBoost      | 56.0      | 56.6              | 57.3              | **57.9**              | 56.0
            | GaussianNB    | 58.1      | 57.7              | 57.2              | **59.3**              | 59.1
            | QDA           | 56.6      | 56.5              | 57.1              | **57.8**              | 56.6
            
            Mean Accuracy Result (w/ opponent's score data) for decision_o
            ----------------------------
            |Model          | w/o Scaler| StandardScaler    | MinMaxScaler      | QuantileTransformer   | PowerTransformer  |
            |:----:         | :----:    | :----:            | :----:            | :----:                |:----:
            | KNN(k=8)      | 54.8      | 57.0              | **58.0**          | 54.6                  | 56.7
            | LR            | 59.5      | **60.2**          | 60.1              | 59.0                  | 59.7
            | Linear SVM    | 57.8      | 59.5              | 60.0              | 58.9                  | **60.2**
            | RBF SVM       | 57.4      | 58.4              | 58.6              | **59.8**              | 58.3
            | DecisionTree  | 56.6      | **57.6**          | 56.9              | **57.6**              | **57.6**
            | RF            | **58.9**  | 57.6              | 57.8              | 58.0                  | 56.9
            | MLP(5,2)      | 57.3      | 58.0              | 58.6              | **59.7**              | 57.1
            | AdaBoost      | 55.9      | 57.3              | 57.5              | 58.0                  | **58.2**
            | GaussianNB    | 57.5      | 57.8              | 56.5              | **58.8**              | 58.5
            | QDA           | **56.2**  | 56.2              | **56.2**          | 57.3                  | 57.4
            
            
             "Linear Regression",
    "LAsso",
    "Ridge",
    "SGDRegressor",
    "RandomForestRegressor",
    "MLPRegressor"
    
             RMSE Result (w/o opponent's score data) for like_o
            ----------------------------
            |Model              | w/o Scaler| StandardScaler    | MinMaxScaler      | QuantileTransformer   | PowerTransformer  |
            |:----:             | :----:    | :----:            | :----:            | :----:                |:----:
            | Linear Regression | 1.82      | 1.82              | 1.82              | 1.81                  | 1.81
            | LAsso             | 1.83      | 1.83              | 1.83              | 1.83                  | 1.83
            | Ridge             | 1.82      | 1.82              | 1.83              | 1.81                  | 1.81
            | SGDRegressor      | N/A       | 1.82              | 1.85              | 1.81                  | 1.80
            | RandomForest      | 1.94      | 1.99              | 2.36              | 1.96                  | 1.95
            | MLP(5,2)          | 1.82      | 1.83              | 2.10              | 1.81                  | 1.81
            
             RMSE Result (w/ opponent's score data) for like_o
            ----------------------------
            |Model              | w/o Scaler| StandardScaler    | MinMaxScaler      | QuantileTransformer   | PowerTransformer  |
            |:----:             | :----:    | :----:            | :----:            | :----:                |:----:
            | Linear Regression | 1.17      | 1.82              | 1.82              | 1.81                  | 1.81
            | LAsso             | 1.27      | 1.83              | 1.83              | 1.83                  | 1.83
            | Ridge             | 1.17      | 1.82              | 1.83              | 1.81                  | 1.81
            | SGDRegressor      | N/A       | 1.82              | 1.85              | 1.81                  | 1.80
            | RandomForest      | 1.27      | 1.99              | 2.36              | 1.96                  | 1.95
            | MLP(5,2)          | 1.17      | 1.83              | 2.10              | 1.81                  | 1.81
            
            

5 결론 (다같이)\

    - 나를 마음에 들어하는 이성을 맞춤 소개시켜줄 수 있는 알고리즘으로의 확장
    
    - 상대방이 나를 마음에 들게 하려면 어떤 점을 개선해야 할지 컨설팅
    
    - 변수를 더 여러개를 써서 성능 향상



