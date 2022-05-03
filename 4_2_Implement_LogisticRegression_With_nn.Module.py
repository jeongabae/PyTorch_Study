#비용 함수 수식에서 가설은 이제 H(x)=Wx+b가 아니라 H(x)=sigmoid(Wx+b)입니다.
#파이토치에서는 nn.Sigmoid()를 통해서 시그모이드 함수를 구현하므로
# 결과적으로 nn.Linear()의 결과를 nn.Sigmoid()를 거치게하면 로지스틱 회귀의 가설식이 됩니다.

#1. 파이토치의 nn.Linear와 nn.Sigmoid로 로지스틱 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

#훈련 데이터를 텐서로 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#nn.Sequential()은 nn.Module 층을 차례로 쌓을 수 있도록함
#nn.Sequential()은 Wx+b와 같은 수식과 시그모이드 함수 등과 같은 여러 함수들을 연결해주는 역할
model = nn.Sequential(
   nn.Linear(2, 1), # input_dim = 2, output_dim = 1
   nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
)