'''
构造kNN分类器
'''

from numpy import *
import operator  #运算符模块
# 创建数据集和标签

group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']

#构建分类器
def classify0(inX, dataSet, labels, k):
    #用于分类的输入向量是inX，输入的训练样本集为dataSet，标签向量为labels，最后的参数k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  # shape[0]是读取矩阵第一维度的长度，即数据的条数
    # 计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #将目标复制成n行，计算得目标与每个训练数值之间的数值之差。
    sqDiffMat = diffMat**2  #各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)  #对应行的平方相加，即得到了距离的平方和
    distances = sqDistances**0.5  #开根号，得到距离
    #排序，确定前k个距离最小元素所在的主要分类
    sortedDistIndicies = distances.argsort()  #argsort函数返回的是数组值从小到大的索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #给字典赋值，每个标签计数
    #返回发生频率最高的元素标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

print("该数据的类别是" + classify0([0, 0],group, labels, 3))