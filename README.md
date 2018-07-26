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
            
            Mean Accuracy Result
            ----------------------------
            |Model | w/o Scaler| StandardScaler | MinMaxScaler | QuantileTransformer |
            |:----:| :----:| :----:| :----:| :----:|
            | KNN(k=8) | 56.5 | **58.4** | 57.9 | 57.8 |
            | LR | 59.9 | 59.9 | 60.3 | **60.6** |
            | Linear SVM | 59.9 | 60.0 | 59.9 | **60.5** |
            | RBF SVM | 58.3 | 59.3 | 58.5 | **59.6** |
            | DecisionTree | 53.1 | 55.8 | 55.8 | **56.0** |
            | RF | **58.2** | 57.5 | 58.1 | 57.3 |
            | MLP(5,2) | 58.1 | 57.4 | **58.9** | 58.3 |
            | AdaBoost | 58.5 | 58.5 | 58.5 | **58.6** |
            | GaussianNB | **58.9** | 58.8 | 58.8 | 58.7 |
            | QDA | **58.8** | **58.8** | **58.8** | 57.0 |
            

5 결론 (다같이)

    - 나를 마음에 들어하는 이성을 맞춤 소개시켜줄 수 있는 알고리즘으로의 확장
    
    - 상대방이 나를 마음에 들게 하려면 어떤 점을 개선해야 할지 컨설팅
    
    - 변수를 더 여러개를 써서 성능 향상



