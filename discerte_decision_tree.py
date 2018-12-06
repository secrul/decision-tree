from math import log
import operator
import numpy as np

"""
从文件读入数据，返回一个二维数组
参数：filename 文件路径
"""
def data_input(filename):
    r_train = open(filename,'r')
    data = r_train.readlines()
    data_x = []
    for line in data:
        line = line.strip('\n').split(' ')
        data_x.append(line)
    return data_x
"""
计算数据集的熵值，通过统计标签的个数和种类或者每种标签的比例
参数 dataSet 数据集
"""
def calcShannonEnt(dataSet):
    #数据集行数
    numEntries=len(dataSet)
    #保存每个标签（label）出现次数的字典
    labelCounts={}
    #对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel=featVec[-1]
        #提取标签信息
        if currentLabel not in labelCounts.keys():#如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1 #label计数
    shannonEnt=0.0 #经验熵
    #计算经验熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #选择该标签的概率
        shannonEnt-=prob*log(prob,2) #利用公式计算
    return shannonEnt #返回经验熵
"""
从原数据集获得子数据集，不包括axis对应的数据
参数 dataSet 原始数据集
    axis 划分属性
    value 划分属性的划分值
"""
def splitDataSet(dataSet,axis,value):
    #创建返回的数据集列表
    retDataSet=[]
    #遍历数据集
    for featVec in dataSet:
        if featVec[axis]==value: #去掉axis特征
             reduceFeatVec=featVec[:axis]
             #将符合条件的添加到返回的数据集
             reduceFeatVec.extend(featVec[axis+1:])
             retDataSet.append(reduceFeatVec) #返回划分后的数据集
    return retDataSet
"""
选出信息增益和信息增益率最大的属性进行数据集的划分
参数 dataSet 数据集
"""
def chooseBestFeatureToSplit(dataSet):
    #特征数量
    info_a = []
    info_a_rate = []
    numFeatures = len(dataSet[0]) - 1
    #计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #信息增益
    bestInfoGain_rate = 0.0
    #最优特征的索引值
    bestFeature = -1
    #遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        #创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        #经验条件熵
        newEntropy = 0.0
        ha = 0
        #计算信息增益
        for value in uniqueVals:
            #subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            #计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            #根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))
            if prob == 1:
                ha = 1
            else:
                ha -= prob*log(prob,2)
        #信息增益
        infoGain = baseEntropy - newEntropy
        infoGain_rate = infoGain / ha#######计算熵值增益率
        #计算信息增益
        info_a.append(infoGain)
        info_a_rate.append(infoGain_rate)
    m = np.mean(info_a)
    for i in range(numFeatures):
        if info_a[i] > m:
            if info_a_rate[i] > bestInfoGain_rate:
                bestInfoGain_rate = info_a_rate[i]
                bestFeature = i
    return bestFeature
"""
统计当前标签集中元素个数最多的标签
参数：classList当前标签集
"""
def majorityCnt(classList):
    classCount={}
    #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    #根据字典的值降序排列
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
"""
建立决策树
参数 dataSet 训练数据集
    labels 训练数据集属性的标签
    featLabels 储存决策树的非叶子节点
"""
def createTree(dataSet,labels,featLabels):
    #取分类标签（是否放贷：yes or no）
    classList=[example[-1] for example in dataSet]
    #如果类别完全相同，则停止继续划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #选择最优特征
    bestFeat=chooseBestFeatureToSplit(dataSet)
    #最优特征的标签
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    #根据最优特征的标签生成树
    myTree={bestFeatLabel:{}}
    #删除已经使用的特征标签
    t_label = labels[:]
    del(t_label[bestFeat])
    #得到训练集中所有最优特征的属性值
    featValues=[example[bestFeat] for example in dataSet]
    #去掉重复的属性值
    uniqueVls=set(featValues)
    #遍历特征，创建决策树
    for value in uniqueVls:
        #统计每个属性地取值个数，如果出现分支少于取值个数，就随机填标签
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value), t_label,featLabels)
    return myTree
"""
根据决策树对训练数据进行预测
参数 inputTree 决策树
    labels 属性的标签集，来获得当前决策树节点对应的属性
    testVec 测试数据
"""
def classify(inputTree,labels,testVec):
    #获取决策树节点
    firstStr=next(iter(inputTree))#节点标签
    #下一个字典
    secondDict=inputTree[firstStr]#子树
    featIndex=labels.index(firstStr)#节点标签序号
    classLabel=[]
    for key in secondDict.keys():
        if testVec[featIndex]==key:#找寻适合分支
            if type(secondDict[key]).__name__=='dict':#仍然有子树
                 classLabel=classify(secondDict[key],labels,testVec)
            else:
             classLabel=secondDict[key]
    return classLabel

if __name__=='__main__':
        file_name = r"C:\Users\liuji\Desktop\third\知识分析\决策树\决策树_data\golf_train.Z7nJ667n.txt"
        file_tname = r"C:\Users\liuji\Desktop\third\知识分析\决策树\决策树_data\golf_test.6meJ6p3T.txt"
        labels = ['a', 'b', 'c', 'd']
        dataSet = data_input(file_name)
        featLabels=[]
        myTree=createTree(dataSet,labels,featLabels)
        #测试数据
        data_test = data_input(file_tname)
        print(myTree)
        ans = []
        for i in range(len(data_test)):
           ans.append(classify(myTree,labels,data_test[i]))
           print(ans[i])
