CIFAR-10 이미지 분류: K-Nearest Neighbors

MinhwanNoh 2022113600

This is an assignment using knn with Cifar-10 dataset.

1. 프로젝트 개요

CIFAR-10을 K-Nearest Neighbors (KNN) 알고리즘을 사용하여 분류하는 과제

**5-Fold 교차 검증(Cross-Validation)**을 적용하여 모델의 하이퍼파라미터를 최적화하고, 다양한 평가지표를 통해 성능을 객관적으로 측정하는 것을 목표로 함

데이터셋: CIFAR-10 (Training 50,000장, Test 300,000장)

주요 알고리즘: K-Nearest Neighbors 

핵심 기술: Scikit-learn Pipeline, PCA, GridSearchCV

2. 프로젝트 수행 과정

가. 데이터 전처리

모든 이미지를 32x32x3 크기에서 3072차원의 1차원 벡터로 Flatten 

각 픽셀 값을 255로 나누어 0~1 사이로 정규화

나. 모델 파이프라인 구축

StandardScaler, PCA, KNeighborsClassifier를 Pipeline으로 연결하여 데이터 스케일링, 차원 축소, 모델 학습 과정을 자동화

PCA : 3072차원의 고차원 데이터를 256차원으로 축소하여 계산 효율성을 높이고 '차원의 저주' 문제를 완화

다. 5-Fold 교차 검증을 통한 K값 최적화

GridSearchCV를 사용하여 5-Fold 교차 검증을 수행. 이를 통해 K값의 변화에 따른 모델의 성능을 안정적으로 평가하고 최적의 K값을 탐색.

탐색 K값 범위: [1, 3, 5, 7, 9, 11, 13, 15]

평가 지표: Accuracy

3. 실험 결과

가. K값에 따른 5-Fold CV 성능 변화

K값이 증가함에 따라 정확도가 꾸준히 상승하는 경향을 보였으며, K=15일 때 평균 정확도가 약 0.5023으로 가장 높게 측정됨

나. 최적 모델 상세 평가지표

교차 검증을 통해 찾은 최적의 모델(K=15)에 대해 cross_val_predict 함수를 사용하여 상세 평가지표를 산출한 결과는 다음과 같음

              precision    recall  f1-score   support

    airplane       0.45      0.68      0.54      5000
  automobile       0.58      0.64      0.61      5000
        bird       0.38      0.50      0.43      5000
         cat       0.35      0.33      0.34      5000
        deer       0.46      0.41      0.44      5000
         dog       0.45      0.38      0.41      5000
        frog       0.62      0.46      0.53      5000
       horse       0.64      0.56      0.59      5000
        ship       0.56      0.70      0.62      5000
       truck       0.70      0.36      0.47      5000

    accuracy                           0.50      50000
   macro avg       0.52      0.50      0.50      50000
weighted avg       0.52      0.50      0.50      50000


전체 정확도 (Accuracy): 약 50%

분석: ship(0.62), automobile(0.61), horse(0.59) 클래스에서 높은 F1-score를 기록하며 비교적 분류 성능이 좋았음. 반면, 형태적으로 유사한 동물 클래스인 cat(0.34), dog(0.41), bird(0.43)는 낮은 F1-score를 보여 분류에 어려움을 겪음 

4. 파일 설명

knn.ipynb: 데이터 로딩부터 교차 검증, 결과 분석 및 최종 예측까지의 모든 과정을 담고 있는 Jupyter Notebook 파일.

best_knn_model_cv.pkl: 5-Fold 교차 검증을 통해 찾은 최적의 하이퍼파라미터로 학습된 최종 모델

submission_cv.csv: 위 모델을 사용하여 테스트 데이터 300,000장을 예측한 최종 제출 파일