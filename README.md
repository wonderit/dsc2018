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
            ```
            ### 1. model : RandomForestClassifier
            Mean Accuracy: 0.573770128235
            ### 2. model : LogisticRegression
            Mean Accuracy: 0.598620230806
            ### 3. model : SVC
            Mean Accuracy: 0.567413843631
            ### 4. model : KNeighborsClassifier
            Mean Accuracy: 0.565009148297
            ### 5. model : DecisionTree
            Mean Accuracy: 0.520274765748
            ```

5 결론 (다같이)

    - 나를 마음에 들어하는 이성을 맞춤 소개시켜줄 수 있는 알고리즘으로의 확장
    
    - 상대방이 나를 마음에 들게 하려면 어떤 점을 개선해야 할지 컨설팅
    
    - 변수를 더 여러개를 써서 성능 향상



