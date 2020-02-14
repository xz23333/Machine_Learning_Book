# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:10:18 2020
项目一：优化约会网站的配对效果
@author: xz
"""

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


# 构建分类器
def classify0(inX, dataSet, labels, k):
    # 用于分类的输入向量是inX，输入的训练样本集为dataSet，标签向量为labels，最后的参数k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  # shape[0]是读取矩阵第一维度的长度，即数据的条数
    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # 将目标复制成n行，计算得目标与每个训练数值之间的数值之差。
    sqDiffMat = diffMat**2  # 各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)  # 对应行的平方相加，即得到了距离的平方和
    distances = sqDistances**0.5  # 开根号，得到距离
    # 排序，确定前k个距离最小元素所在的主要分类
    sortedDistIndicies = distances.argsort()  # argsort函数返回的是数组值从小到大的索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 返回发生频率最高的元素标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 解析文本文件为Numpy
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建以0填充的numpy矩阵
    returnMat = zeros((numberOfLines, 3))  # 生成一个 n*3(n行3列的)的矩阵，各个位置上全是 0
    classLabelVector = [] # 返回的分类标签向量
    index = 0 # 行的索引值
    # 解析文件数据到列表
    for line in arrayOLines:
        # 按行读取数据，strip()去除首尾的空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 将每一行的字符串根据'\t'分隔符进行切片，获得元素列表
        listFromLine = line.split('\t')
        # 选取前3个元素,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # -1表示最后一列，最后一列是类别，将其存储到向量classLabelVector中。
        # 告诉解释器存储的元素值为整型，否则会当作字符串处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回特征矩阵和分类标签向量
    return returnMat, classLabelVector

def drawing(datingLabels, datingDataMat):
    LabelsColors = []
    # 设置颜色
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    fig = plt.figure()
    ax = fig.add_subplot(221)
    # 画出以第一、第二列数据为轴的散点分布图
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1], color=LabelsColors, s=15, alpha=.5)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel("每年获得的飞行常客里程数")
    plt.ylabel("玩视频游戏所耗时间百分比")

    bx = fig.add_subplot(222)
    bx.scatter(datingDataMat[:,0],datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    plt.xlabel("每年获得的飞行常客里程数")
    plt.ylabel("每周消费的冰淇淋公升数")

    ax = fig.add_subplot(223)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    plt.xlabel("玩视频游戏所耗时间百分比")
    plt.ylabel("每周消费的冰淇淋公升数")
    plt.show()


# 数值归一化
def autoNorm(dataSet):
    # 归一化公式：Y = (X-Xmin)/(Xmax-Xmin)
    # min和max是1*3的矩阵，存着每一个特征的最值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals # 极差
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 读取矩阵第一维度的长度
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#验证分类器，计算错误率
def datingClassTest():
    hoRatio = 0.10  # 取数据集中的10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    drawing(datingLabels, datingDataMat)
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 获得normMat的行数
    numTestVecs = int(m*hoRatio)  #10%的数量
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("预测数值为 %d, 实际数值为: %d"
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率为 %f" % (errorCount/float(numTestVecs)))


datingClassTest()

#预测
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", resultList[classifierResult-1])

classifyPerson()