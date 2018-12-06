import numpy as np
from math import log
import operator
import copy
"""
读入数据
删除数据集的第一列
返回二维数组
"""
def data_input(filename):
    r_train = open(filename,'r')
    data = r_train.readlines()
    data_x = []
    for line in data:
        line = line.strip('\n').split(' ')
        del line[0]
        tem = [float(x) for x in line]
        data_x.append(tem)
    return data_x
"""
如果相应属性值小于分界值value，则划分为左子树
返回子数据集
参数 dataset 数据集
    axis 对应的属性序号
    value 分界值
"""
def splitDataleft(dataset,axis,value):
    #连续值保留当前属性，可能再次作为标准划分
    sub_dataset = []
    for row in dataset:
        if row[axis] < value:
            sub_dataset.append(row)
    return sub_dataset
"""
如果相应属性值大于分界值value，则划分为右子树
返回子数据集
参数 dataset 数据集
    axis 对应的属性序号
    value 分界值
"""
def splitDataright(dataset,axis,value):
    #连续值保留当前属性，可能再次作为标准划分
    sub_dataset = []
    for row in dataset:
        if row[axis] > value:
            sub_dataset.append(row)
    return sub_dataset
"""
计算熵值 统计0/1所占比例
参数 dataset 数据集
返回浮点数 熵值
"""
def cal_shang(dataset):#标签其实只有0/1两种
    num = len(dataset)
    labelcounts = {}
    for row in dataset:
        tem_label = row[-1]
        if tem_label not in labelcounts.keys():
            labelcounts[tem_label] = 0
        labelcounts[tem_label] += 1

    p = 0
    for key in labelcounts:
        p = float(labelcounts[key]) / num
        break
    gini = 2 * p * (1 - p)
    return gini
"""
从一个属性的n-1个可能熵值中找一个最大的熵值及分界数
返回 最大的熵值及其对应的分界数据
参数 sunique 去重排序后一个属性的数据
    dataset 数据集
    i 属性的序号
"""
def best_col_flag(sunique,dataset,i):
    length = len(sunique) - 1
    best_m = 0
    best_gini = 1000
    for j in range(length):  # 该属性中地取值范围
        m = (sunique[j] + sunique[j + 1]) / 2
        subDataSet1 = splitDataleft(dataset, i, m)
        # 计算子集的概率
        p1 = len(subDataSet1) / float(len(dataset))
        # 根据公式计算经验条件熵
        newgini = p1 * cal_shang(subDataSet1)
        subDataSet2 = splitDataright(dataset, i, m)
        # 计算子集的概率
        p2 = len(subDataSet2) / float(len(dataset))
        # 根据公式计算经验条件熵
        newgini += p2 * cal_shang(subDataSet2)
        if best_gini > newgini:#找出最小的GINI
            best_m = m
            best_gini = newgini
    return best_gini,best_m
"""
从n个属性对应的n个熵值中选出最大的熵值对应属性序号还有分界数据
返回 属性序号和分界数
参数 dataset 数据集
"""
def choosebest(dataset):
    bestaxis = -1#最大熵值对应的属性序号
    bb_gini = 10000 #最大的熵值
    num = len(dataset[0]) - 1#最后一列是标签
    best_m = 0
    for i in range(num):#循环取出原数据集的一列，找出最佳地分类点
        col = [row[i] for row in dataset]
        unique = set(col)#数据去重
        suniquq = sorted(unique)#数据从大到小排序
        best_gini,best_num = best_col_flag(suniquq,dataset,i)#一列中最大的熵值和分界数
        if best_gini < bb_gini:#记录最大的信息增益和对应属性序号和分界数
            bb_gini = best_gini
            bestaxis = i
            best_m = best_num
    return bestaxis,best_m
"""
建立决策树
参数 dataset 训练数据集
    labels 数据集属性的标签
    inode 记录节点
"""
def creatTree(dataset,labels,inode,ct):
    print(len(dataset))
    if len(dataset) == 0:#如果数据集已经空了返回1标签
        return 0
    if ct > 150:
        return 1
    classlist = [row[-1] for row in dataset]
    if classlist.count(classlist[0]) == len(classlist):#数据集标签已经相同
        return classlist[0]

    best_axis,best_m= choosebest(dataset)#最佳的属性和分类数
    best_label = labels[best_axis]#最佳属性对应的label
    inode.append(best_label)
    myTree = {best_label:{}}#建立决策树
    #递归处理左子树
    myTree[best_label][-best_m] = creatTree(splitDataleft(dataset,best_axis,best_m),labels,inode,ct+1)
    #处理右子树
    myTree[best_label][best_m] = creatTree(splitDataright(dataset,best_axis,best_m),labels,inode,ct+1)
    return myTree
"""
按照决策树对测试集进行预测
返回标签 0/1
参数 Tree 建立的决策树
    labels 判断决策树节点对应的属性
    test 测试数据
"""
def classify(Tree,labels,test):
    if type(Tree).__name__ != 'dict':#如果决策树已经到达叶节点
       return Tree
    first = next(iter(Tree))#取出根节点对应的key
    second = Tree[first]#对应的value
    inode_index = labels.index(first)#根节点标签对应属性的序号
    classlabel = []
    for key in second.keys():#遍历根节点对应的value，如果大于右否则左子树子树
        if test[inode_index] < abs(key) and key < 0:
            if type(second).__name__ == 'dict':
                classlabel=classify(second[key],labels,test)
            else:
                classlabel = second[key]
        if test[inode_index] > abs(key) and key > 0:
            if type(second).__name__ == 'dict':
                classlabel=classify(second[key],labels,test)
            else:
                classlabel = second[key]
    return classlabel

if __name__ == '__main__':
    file_name = r"C:\Users\liuji\Desktop\third\知识分析\决策树\决策树_data\train.txt"
    file_tname = r"C:\Users\liuji\Desktop\third\知识分析\决策树\决策树_data\test.txt"
    data_set = data_input(file_name)
    data_test = data_input(file_tname)
    ans = []
    labels = []
    length = len(data_test[0])
    for i in range(length):
        labels.append(str(i))
    inode_index = []
    ct = 0
    myTree = creatTree(data_set,labels,inode_index,ct)
    print(myTree)#打印决策树
    for i in range(len(data_test)):
       ans.append(classify(myTree,labels,data_test[i]))
    te = []
    for x in ans:#转换为整型，并对缺失数据填补
        t = -1
        if x == 0.0:
            t = 0
        else:
            t = 1
        print(t)
        te.append(t)

    with open(r"D:\ans.txt",'w') as f1:#输出到文件
        for x in te:
            f1.write(str(x))
            f1.write('\n')
