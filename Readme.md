### 🦸‍♂️ MASK ,Age, Gender Classification

##### 본프로젝트는 🚀NAVER AI BoostCamp에서 개최한 competition입니다
#####📆 2021.03.29~2021.04.09
---
### 🥇 최종결과

##### 🧨 Public Score: 0.7835 4등 (240명)
##### 🏆 Private Score: 0.7726 3등 

---
### ✏ 문제 정의 및 해결 방법

- 해당 문제를 풀기위해 어떻게 접근하고, 구현하고, 최종적으로 사용한 솔루션에 대해서는 [report](https://songbae.oopy.io/2c675fb0-30b9-4f7e-99f6-1ff2e11971b3)에서 확인 할 수 있습니다
- 위 Report는 개인의 회고도 포함하고 있습니다.

---

### Code 구조

```
C:.
│  loss.py                                  ### loss function 정의
│  Readme.md                                
│  test.py                                  ### inference code                    
│  train.py                                 ### train code
├─config
│      config.py                            ### config code -> args 를 통한 수정 가능
│      __init__.py
├─data
│  │  dataset.py                            ### data 전처리 및 이미지 data를 train하기 전까지 과정
│
├─ETC
│      crop_test_img.ipynb                  ### image crop test-> detectron model을 이용하여 image crop
│    
│      show_image.ipynb                     ### image 출력 -> EDA를 통한 이미지 특징 파악
│      
│    
│
├─models
│      focal_loss.py                        ### focal loss 구현
│      __init__.py
│
├─utils
│      view_model.py                        ### 모델 
│
└─__pycache__
        loss.cpython-37.pyc

```
### 💡 MASK | AGE | Gender Classification

- **Competition Data 개요**
    1.  전체 사람 명 수 : 4500
    2.  한 사람당 사진의 개수 : 7( 마스크 착용 6, 이상하게 착용(턱스크,코스크) 1장, 미착용 1장 ) 
    3.  이미지 크기: (384,512) 
    4.  Evaluation : F1_score (Macro) 
   ### EDA (Exploratory Data Analysis,탐색적 데이터 분석)

   - **성별 분포**
    여성 1600 남성 1100명 정도로 분포하고 있으며 많은 차이는 아니기 때문에 data imbalance 라고 보기엔 어렵다

   - **Age 분포**
        > 0~30 과 30~60 의 비율은 비슷하지만 60대의 비율이 압도적으로 적은 것을 볼 수 있다. 즉 data imbalance가 심하다.
        - **Mask 분포**
            - [5: 1 :1]로 완전히 같은 비율을 가지고 있다.
            - 마스크마다 색이 다른 것도 존재 하며 , 마스크 대신 손수건을 착용한 이미지도 존재
            - 턱스크, 코스크 ,눈스크 등 마스크를 잘 못 착용한 다양한 이미지 존재
            - 마스크를 아예 착용하지 않은 클린한 이미지
            
### 📜 최종 모델 및 파라미터
   ---
   1. Efficient_Net B3 
   2. Learning Rate:
       1. Age Label : 1e-4 
       2. Mask Label: 3e-4 
       3. Gender Label: 1e-4
   3. Optimizer : Adam 
   4. Loss :
       1. Crossentropy_loss: Mask, Gender 
       2. Focal_loss: age
   5. Epoch:
       1. Age: 4 Epoch 
       2. Mask , Gender : 10 Epoch 
   6. Batch_size: 64

### **[전처리]**

---

1.  Facenet_pytorch : MTCNN 을 이용하여 face부분 crop (Train img, Test_img )
2.  MTCNN을 통해 face가 detect 되지 않은 이미지는 Center_Crop [300,300]
3.  Labeling :
    1. Age→ [0~30 : 0 |  30~58:1 | 59~60: 2 | 
    2. Mask → 0 | 1 | 2 
    3. Gender: Female: 0 | Male : 1

### [Augmentation]

---

1. Mask:
    - Perspective, Rotate,RadomBrightness, HueSaturationValue, RandomContrast
2. Age:
    - Rotate, RandomGridshuffle(2,2), Perspective, GaussNoise
3. Gender
    - Rotate, Perspective, GaussNoise
4. ALL→ Resize(224,224) , Normalize()

### [Train & Validation]

---

1. 5_Fold 중 1개의 Fold를 선택하여 Train Set, Valid set 분할 
2. Best F1-Score를 이용하여 model_parameter() update 
3. Seed : 777 고정 
4.  Mask, Age, Gender 별로 다른 Learning Rate, Loss,  augmentation 적용 

### [Environment]

---

- Window 10
- VsCode
- GPU P-40


