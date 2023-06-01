import json
import pandas as pd


def getID(filePath, savePath, col):
    f = pd.read_csv(filePath)
    pointType = f[col]
    TypeList = pointType.tolist()

    count = 0
    labelDict = {}
    for i in TypeList:
        if i not in labelDict.keys():
            labelDict[i] = count
            count += 1

    data = json.dumps(labelDict, ensure_ascii = False)
    with open(savePath, 'w', encoding='utf-8') as fp:
        fp.write(data)


def loadID(path):
    with open(path, 'r', encoding='utf-8') as fp:
        ids = json.load(fp)
    return ids


if __name__ == "__main__":
    getID('../Question.csv', 'labelID.json' ,'point_type')
    ids = loadID('labelID.json')
