import math
def createDataSet():
    dataSet = [
    ["1","杂食动物", True, False, False, True],
    ["2","杂食动物", True, False, False, True],
    ["3","肉食动物", True, False, False, True],
    ["4","肉食动物", False, False, True, False],
    ["5","肉食动物", False, True, False, False],
    ["6","肉食动物", False, False, False, False],
    ["7","杂食动物", True, False, True, True],
    ["8","草食动物", True, False, False, True],
    ["9","杂食动物", False, False, True, False],
    ["10","肉食动物", False, True, False, False],
    ["11","肉食动物", True, True, False, True],
    ["12","肉食动物", False, False, False, True],
    ["13","草食动物", True, False, False, True],
    ["14","肉食动物", False, False, False, False]
    ]
    labels = ['动物','饮食习惯'	,'胎生动物',	'水生动物',	'会飞',	'哺乳动物']   #分类属性
    return dataSet, labels                     #返回数据集和分类属性

dfy=['饮食习惯'	,'胎生动物',	'水生动物',	'会飞']
def h(lista):
    sums = 0
    for i in lista:
        x = i / sum(lista)
        sums += x * (math.log(x, 2))
    return -(sums)
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}                        #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]                #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = h(labelCounts.values())
#     print(list(labelCounts.values()),'--',sum(list(labelCounts.values())))
    return shannonEnt
def calcShannonEntxx(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}                        #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]                #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    se=''
    for i in labelCounts.values():
        se+=f"{i}/{sum(labelCounts.values())}log({i}/{sum(labelCounts.values())})+"
#     print(list(labelCounts.values()),'--',sum(list(labelCounts.values())))
    return se[:-1]



def splitDataSet(dataSet, axis, value):
    retDataSet = []             #创建返回的数据集列表
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]        #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:]) #将符合条件的添加到返回数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet     #返回划分后的数据集


def chooseBestFeatureToSplit(dataSet,bestFeatLabel):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    vb = []
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 经验条件熵
        # 计算信息增益
        aum = 0
        lok=[]
        for j in uniqueVals:
            x = splitDataSet(dataSet, i, j)
            l = len(x)
            aum += calcShannonEnt(x) * (l / len(dataSet))
            # print(f'{dfy[i]}|{bestFeatLabel}={l}/{len(dataSet)}*{ calcShannonEntxx(x)}')
            print()
            lok.append((l / len(dataSet)))
        lnumn = 0.000000000001
        for ii in lok:
            lnumn += ii * math.log(ii, 2)
        lnumn = -lnumn
        # print(f"H({dfy[i]}|{bestFeatLabel})特征的增益为:{(baseEntropy)}-{aum}={round(baseEntropy - aum, 3)}")
        # print(f"H({dfy[i]}|{bestFeatLabel})特征的增益率为:{round(baseEntropy - aum, 3)}/{lnumn}={round((baseEntropy - aum)/lnumn,5)}")
        vb.append((baseEntropy - aum)/lnumn)
        print('------------')
    if not vb:
        return -1
    bestInfoGain = vb.index(max(vb))
    print("============")
    return bestInfoGain

print(calcShannonEntxx(createDataSet()[0]))
def createTree(dataSet, labels, featLabels,str):
    classList = set([example[-1] for example in dataSet])

    if len(labels) == 0 or len(classList) == 1:
        return majorityClass(dataSet)

    bestFeat = chooseBestFeatureToSplit(dataSet,str)
    if bestFeat == -1:
        return majorityClass(dataSet)
    bestFeatLabel = labels[bestFeat]
    del (labels[bestFeat])
    del (dfy[bestFeat])
    featLabels.append(bestFeatLabel)
    featValues = set([example[bestFeat] for example in dataSet])
    disct = {}

    for value in featValues:
        subLabels = labels.copy()
        disct[value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels,bestFeatLabel)

    return {bestFeatLabel: disct}


def majorityClass(dataSet):
    classCount = {}
    for example in dataSet:
        classCount[example[-1]] = classCount.get(example[-1], 0) + 1
    return max(classCount, key=classCount.get)


if __name__ == '__main__':
    data = {}
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels, [],'')

    print(myTree)