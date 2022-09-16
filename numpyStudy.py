import numpy as np

# numpy
# 과학 계산을 위해 만들어진 라이브러리
# 행렬/배열 처리 및 연산이 가능
# 난수 생성등 여러 가지 메소드 사용 가능

# a는 파이썬의 리스트(list)
a = [1, 2, 3]
print(a)

# b는 numpy의 배열(array)
b = np.array(a)
print(b)

# numpy의 배열은 numpy 라이브러리 내 ndarray 라는 클래스
print(type(b))

# ndarray 클래스 정보 조회
print(dir(b))

# 객체에 대한 도움말 보기
help(b)

# 특정 메소드에 대한 도움말 보기
help(b.fill)

b.fill(0)
print(b)


# 리스트와 배열의 차이
anythings = [1, 0.2, '삼']  # 내부 요소들의 타입을 신경쓰지 않는다.
print(anythings)

anyArray = np.array(anythings)
print(anyArray)  # 내부 요소들의 타입이 통일되어야 한다.


# 리스트와 배열의 선언

myList = [];

myArray = np.array([])

print(myList)
print(myArray)

print(type(myList))
print(type(myArray))


# 값을 부여하며 선언하기

myList = [1, 2, 3]
myArray = np.array([1, 2, 3])

print(myList)
print(myArray)


# 값 초기화

# 메소드를 이용하거나, 빈 값으로 초기화
myList.clear()
myList = []
print(myList)

myArray = np.array([]);
print(myArray)



# 리스트와 배열에 값 추가

myList.append(1)
myList.append(2)
myList.append(3)

print(myList)

# append에 커서 놓고 shift + tab 누를 시 도움말 표시(Jupyter Notebook 기준)
myArray = np.append(myArray, 1)
myArray = np.append(myArray, 2)
myArray = np.append(myArray, 3)

print(myArray)


# 배열 내 요소들의 타입 확인
print(myArray.dtype)


# 리스트 내 타입 변경
myArray = myArray.astype(int)


print(myArray.dtype)
print(type(myArray[1]))  # 배열 내 요소는 전부 타입이 같으므로 하나의 데이터의 타입을 조회해도 된다.


# 리스트와 배열 내 요소 조회(인덱싱)

print(myList[1])
print(myArray[1])

# 인덱스에 음수를 붙여 거꾸로 접근이 가능
print(myArray[-1])
print(myArray[-2])

# 콜론(:) 이용
# [start:end] start 인덱스부터 end-1 인덱스까지 출력
print(myArray[0:2])
print(myArray[:])   # 생략시 전체를 의미한다. (start 는 0, end 는 myArray의 길이 = [0:3])
print(myArray[1:])  # 인덱스 1부터 나머지 전체 (= [1:3])
print(myArray[:2])  # 인덱스 0부터 1까지 (= [0:2])


# 더블 콜론(::) 이용
# [start : end : step] start 인덱스부터 end-1 인덱스까지 step 씩 뛰어가며 출력
print(myArray[0:3:1]) # 인덱스 0, 1, 2 출력
print(myArray[0:3:2]) # 인덱스 0, 2 출력
print(myArray[::-1])  # 배열을 거꾸로 리턴



# 리스트와 배열 내 요소 수정

myList[1] = 30
myArray[1] = 30

print(myList)
print(myArray)


# 리스트와 배열 내 값 삭제

# 리스트
# remove() : 괄호 안 값에 해당하는 데이터를 삭제, 값이 없다면 에러 발생
# 파이썬에서의 에러 처리
try:
    myList.remove(2)  # 에러가 발생할 수 있는 코드
except:
    pass  # 에러 발생시 넘어가도록 처리
print(myList)

# pop() : 괄호 안 인덱스에 해당하는 데이터를 삭제, 해당 인덱스가 존재하지 않으면 에러 발생
myList.pop(0)
print(myList)


# 배열
# np.delete(arr, index) : 배열(arr) 내 index에 해당하는 요소들을 제거, 해당 인덱스가 존재하지 않으면 에러 발생
myArray = np.delete(myArray, 1)
print(myArray)

# 값을 지정하여 제거하기 위해서는 np.where()를 이용
# np.where(조건) : 조건에 충족하는 요소의 인덱스를 리턴한다, 해당 값이 존재하지 않아도 에러가 발생하지 않음
myArray = np.delete(myArray, np.where(myArray == 5))
print(myArray)


# for문을 이용하여 리스트와 배열에 값 추가

for i in range(0, 10):
    myList.append(i)
    myArray = np.append(myArray, i)

print(myList)
print(myArray)

# 리스트와 배열 내 데이터 처리 (for문)

# 리스트와 배열의 길이를 이용한 for문
print(len(myList))
print(len(myArray))
for i in range(0, len(myList)):
    print('list index ' + str(i) + " value = " + str(myList[i]))

print("===========================")

for i in range(0, len(myArray)):
    print('array index ' + str(i) + " value = " + str(myArray[i]))


# 0부터 9까지 순차적으로 나열된 리스트 및 배열 얻기
print(range(0, 10))
print(type(range(0, 10)))
print(list(range(0, 10)))

print(np.arange(0, 10))

# 9부터 0까지 순차적으로 나열된 리스트 및 배열 얻기
print(list(range(9, -1, -1)))
print(np.arange(9, -1, -1))

# 리스트와 배열 자체를 이용한 for문(foreach)
for i in myList:
    print(i)  # 여기서 i는 리스트의 내부 요소이다

print("===========================")

for i in myArray:
    print(i)

# 홀수값을 전부 제거
# 인덱스 접근 후 홀수면 제거

# 랜덤숫자 10개를 myArray에 추가
myArray = np.array([]);
for i in range(0, 10):
    myArray = np.append(myArray, int(np.random.rand() * 50))

print(myArray)

"""
for i in range(0, len(myArray)):
    if myArray[i] % 2 == 1:
        myArray = np.delete(myArray, i)  # 앞에서부터 접근해서 제거하면 인덱스가 앞당겨지므로 문제가 발생한다.

print(myArray)
"""
# 거꾸로 접근해나가며 제거하는 방식
for i in range(len(myArray) - 1, -1, -1):
    if myArray[i] % 2 == 1:
        myArray = np.delete(myArray, i)

print(myArray)

# 랜덤숫자 10개를 myArray에 추가
myArray = np.array([]);
for i in range(0, 10):
    myArray = np.append(myArray, int(np.random.rand() * 50))

print('myArray = ' + str(myArray))

# 제거할 대상들의 인덱스를 따로 저장 후, np.delete()로 한번에 제거
idxList = []
for i in range(0, len(myArray)):
    if myArray[i] % 2 == 1:
        idxList.append(i)

print(idxList)  # 홀수에 해당하는 인덱스들을 저장한 리스트

myArray = np.delete(myArray, idxList)  # np.delete() 두번째 파라미터에 인덱스 배열을 넣으며 해당 인덱스에 대한 데이터 삭제
print('myArray = ' + str(myArray))



# 배열 내에 최댓값 찾기

print(myArray)
maxVal = myArray[0]  # 0으로 두기 보다는 배열의 인덱스 0번째 요소를 두는 것이 좋다

for i in myArray:
    if maxVal < i:
        maxVal = i

print('최댓값: ' + str(int(maxVal)))


# 배열 정렬

# 오름차순 정렬
myArray = np.sort(myArray)
print(myArray)

# 내림차순 정렬
# 오름차순으로 정렬한 후 정렬된 배열을 거꾸로 출력하는 것과 동일
myArray = np.sort(myArray)[::-1]
print(myArray)

# 배열을 잘 다루기 위해 정렬 알고리즘을 사용하여 정렬

# 선택 정렬 (과제)
# 1. 해당 배열에서 최댓값을 찾아 해당 위치의 값과 배열의 마지막 값의 자리를 바꾼다(오름차순 기준).
# 2. 배열의 마지막을 제외한 나머지 값들 중 최댓값을 찾아 해당 위치의 값과 배열의 끝에서 두번째 값을 swap 한다.
# 3. 위 과정을 배열의 길이만큼 반복하면 정렬이 잘 될겁니다.
# * 한번에 정답을 찾으려하지 마시고, 시행착오를 겪어가면서 정답에 접근하시는게 오히려 더 빠를 수 있습니다.
# 제출하실 필요는 없고, 스스로 해보시면 배열 내 요소들을 처리하는 실력이 크게 향상됩니다.

for i in range(0, len(myArray)):
    maxVal = myArray[0]
    maxIdx = 0
    for j in range(0, len(myArray) - i):
        if maxVal < myArray[j]:
            maxIdx = j
    temp = myArray[len(myArray) - i - 1]
    myArray[len(myArray) - i - 1] = myArray[maxIdx]
    myArray[maxIdx] = temp

print(myArray)


# 다차원 배열 처리

# 1차(x축), 2차(x축, y축), 3차(x축, y축, z축), 4차(시공간)
xAxis = np.array([1, 2, 3])
print(xAxis)
xyAxis = np.array([[1, 2, 3], [4, 5, 6]])  # 1행 [1, 2, 3] / 2행 [4, 5, 6] / 1행 1열 [1] / 1행 2열 [2]
print(xyAxis)
print(len(xyAxis)) # 몇행?
print(len(xyAxis[0])) # 몇열?
print(xyAxis.shape) # (몇행, 몇열)
print(xyAxis[0])  # 1행
print(xyAxis[1])  # 2행
print(xyAxis[0][0])  # 1행 1열
print(xyAxis[0][1])  # 1행 2열

xyzAxis = np.array([ [ [1, 2, 3], [4, 5, 6] ], [ [7, 8, 9], [10, 11, 12] ] ])  # 1층 [[1,2,3], [4,5,6]] / 1층 1행 [1,2,3]
print(xyzAxis)
print(xyzAxis[0]) # 1층
print(xyzAxis[1][0])  # 2층 1행
print(xyzAxis[0][1][2])  # 1층 2행 3열 = 6


# 행렬 연산

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A = np.arange(1, 10).reshape(3, 3)  # 1부터 9까지의 요소를 가지는 배열의 모양을 3행 3열로 바꿈

# 행렬 더하기
print(A + 3)

# 행렬 곱하기
print(A * 2)


B = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

# 행렬끼리 더하기 (두 행렬의 행과 열의 수가 같아야 한다)
C = A + B
print(C)

# 행렬끼리 각 자리 곱하기 (두 행렬의 행과 열의 수가 같아야 한다)
D = A * B
print(D)

# 행렬 곱 (A 행렬의 열과 B행렬의 행의 수가 같아야 한다)
E = np.array([[19, 20, 21], [22, 23, 24], [25, 26, 27]])
F = A@E  # 외적
print(F)
G = A.dot(E)  # 내적
print(G)

# 행렬의 전치 연산(Transpose)
H = A.T
print(A)
print(H)

# 분배 법칙
# (A+B)C = AC + BC
I = (A + B) @ C
J = A @ C + B @ C
print(I)
print(J)

# 데이터 가져오기 .csv

import csv

f = open('exchange.csv', 'r')
rdr = csv.reader(f)

dateArray = np.array([])
vesArray = np.array([])
krwArray = np.array([])
jpyArray = np.array([])

for line in rdr:
    dateArray = np.append(dateArray, line[0])
    vesArray = np.append(vesArray, line[1])
    krwArray = np.append(krwArray, line[2])
    jpyArray = np.append(jpyArray, line[3])

f.close()


# 헤더 값 제거

dateArray = np.delete(dateArray, 0)
vesArray = np.delete(vesArray, 0)
krwArray = np.delete(krwArray, 0)
jpyArray = np.delete(jpyArray, 0)

print(dateArray[0])
print(vesArray[0])
print(krwArray[0])
print(jpyArray[0])


print(type(vesArray[0]))

vesArray = vesArray.astype(float)
krwArray = krwArray.astype(float)
jpyArray = jpyArray.astype(float)


# 데이터 그리기(matplotlib)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""
# 그림의 크기 설정
plt.figure(figsize=(20, 10))

# 그리기
# plt.plot(x축 배열, y축 배열, 색, 범례)
plt.plot(dateArray, vesArray, 'r', label="VES")

plt.legend(loc='best', prop={'size': 20})

# 눈금 표시를 위한 배열
xlist = []
for i in range(0, len(dateArray), 20):
    xlist.append(dateArray[i])


# 눈금 표시
plt.xticks(xlist, rotation = 30, fontsize = 15)
plt.yticks(np.arange(np.min(vesArray), np.max(vesArray), np.mean(vesArray)/10), fontsize = 15)

# x축 그리는 범위 설정
plt.xlim(xlist[0], xlist[len(xlist)-1])

plt.show()


plt.figure(figsize=(20, 10))

# 그리기
# plt.plot(x축 배열, y축 배열, 색, 범례)
plt.plot(dateArray, krwArray, 'b', label="KRW")

plt.legend(loc='best', prop={'size': 20})

# 눈금 표시를 위한 배열
xlist = []
for i in range(0, len(dateArray), 20):
    xlist.append(dateArray[i])

# 눈금 표시
plt.xticks(xlist, rotation = 30, fontsize = 15)
plt.yticks(np.arange(np.min(krwArray), np.max(krwArray), 10), fontsize = 15)

# x축 그리는 범위 설정
plt.xlim(xlist[0], xlist[len(xlist)-1])

plt.show()


# 두개 그리기 (normalize 가 안되어있어서 비교하기 힘들다)

plt.figure(figsize=(20, 10))

# 그리기
# plt.plot(x축 배열, y축 배열, 색, 범례)
plt.plot(dateArray, krwArray, 'b', label="KRW")
plt.plot(dateArray, jpyArray, 'g', label="JPY")

plt.legend(loc='best', prop={'size': 20})

# 눈금 표시를 위한 배열
xlist = []
for i in range(0, len(dateArray), 20):
    xlist.append(dateArray[i])

# 눈금 표시
plt.xticks(xlist, rotation = 30, fontsize = 15)
plt.yticks(np.arange(np.min(jpyArray), np.max(krwArray), 100), fontsize = 15)

# x축 그리는 범위 설정
plt.xlim(xlist[0], xlist[len(xlist)-1])

plt.show()
"""

# 데이터 처리

# 각 데이터 normalize 하기 (최대값이 1로 표시되도록 데이터 처리)

exam = np.array([1, 2, 3, 4, 5])
print(exam)

# 단순히 배열에 최댓값을 나눠주면 각각의 요소에 나누기가 적용된다.
exam = exam / 5
print(exam)

vesArray = vesArray / np.max(vesArray)
print(vesArray)

krwArray = krwArray / np.max(krwArray)
jpyArray = jpyArray / np.max(jpyArray)


# 데이터 그리기(matplotlib) - 환율 동향

plt.figure(figsize=(20, 10))

# 그리기
# plt.plot(x축 배열, y축 배열, 색, 범례)
plt.plot(dateArray, vesArray, 'r', label="VES")
plt.plot(dateArray, krwArray, 'b', label="KRW")
plt.plot(dateArray, jpyArray, 'g', label="JPY")

plt.legend(loc='best', prop={'size': 20})

# 눈금 표시를 위한 배열
xlist = []
for i in range(0, len(dateArray), 20):
    xlist.append(dateArray[i])

# 눈금 표시
plt.xticks(xlist, rotation = 30, fontsize = 15)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 15)

# x축 그리는 범위 설정
plt.xlim(xlist[0], xlist[len(xlist)-1])

plt.show()





