from transformers import BertTokenizer, BertModel
from GetID import loadID
import torch
import copy
import json
from typing import *

from BuildTree import TreeNode, buildTree, LCA, TreeNodes

import pandas as pd
from tqdm import tqdm


def str2lst(i):
    return list(map(float, i.split(" ")))


class PrePoint:
    def __init__(self,
                 IDPath="../data/labelID.json",
                 quesTypePath="../data/questionType.json",
                 trainDataPath="../data/train.csv",
                 modelDir="../model_save",
                 catePath="../data/category.json",
                 device="cpu"):
        self.device = device
        train = pd.read_csv(trainDataPath)
        train_x = train["embed"].tolist()
        self.x = torch.tensor([str2lst(str(i)) for i in train_x]).to(device)
        self.y = torch.tensor(train["label_id"].tolist()).to(device)
        self.train_type = torch.tensor(train["question_type"].tolist()).to(device)
        self.prey = []

        self.IDdict = loadID(IDPath)
        self.IDdict = dict(zip(self.IDdict.values(), self.IDdict.keys()))  # {id: type}
        self.typeDict = loadID(quesTypePath)
        self.typeDict = dict(zip(self.typeDict.values(), self.typeDict.keys()))

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-chinese",
                                                       cache_dir=modelDir)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-chinese", cache_dir=modelDir)

        dct = json.load(open(catePath, 'r', encoding='utf-8'))
        self.root = TreeNode("root", 0)
        buildTree(self.root, dct, 1)

    def predict(self, topK: int, examTXT: str, isVote: bool = False, quesType: str = None) -> List[Tuple[str, float]]:
        if len(examTXT) > 510:
            examTXT = examTXT[:510]

        input_ids = self.tokenizer.encode(examTXT, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids)
            outputs = outputs.last_hidden_state[0][0].to(self.device)

        mat = torch.inner(self.x, outputs)

        if quesType is not None:
            self.mask(mat, quesType)

        _, ids = mat.topk(topK, dim=0, sorted=True)
        labels = {}
        if isVote:
            vote = torch.arange(start=1, end=1 + (topK + 1) * 0.1, step=0.1)
            for k, id in enumerate(ids):
                cls = self.y[id].item()
                if self.IDdict[cls] not in labels:
                    labels[self.IDdict[cls]] = vote[k]
                else:
                    labels[self.IDdict[cls]] += vote[k]
            labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
            labels = list(list(zip(*labels))[0])
        else:
            labels = [self.IDdict[self.y[id].item()] for id in ids]

        self.prey.append(labels)  # [[point1, point2, point3, ...], [], ...]
        return labels

    def mask(self, mat, quesType: str):
        for k, i in enumerate(self.train_type):
            if self.typeDict[i.item()] != quesType:
                mat[k] = -1e9

    def topKacc(self, topk: int, depth: int, y: List[int]) -> List[int]:
        self.errId = []
        tot = len(y)
        acc = torch.zeros(topk)
        for j, lst in enumerate(self.prey):
            for k, i in enumerate(lst):
                if self.IDdict[y[j]] not in TreeNodes.keys() or i not in TreeNodes.keys():
                    continue
                if depth == -1:
                    if TreeNodes[i] == TreeNodes[self.IDdict[y[j]]]:
                        acc[k] += 1
                        break
                else:
                    if LCA(TreeNodes[i], TreeNodes[self.IDdict[y[j]]]).depth > depth:
                        acc[k] += 1
                        break
                    else:
                        if self.errId == [] or self.errId[-1] != j:
                            self.errId.append(j)

        for i in range(1, topk):
            acc[i] += acc[i - 1]
        acc /= tot
        return acc


if __name__ == "__main__":
    pre = PrePoint(device="cuda")

    f = pd.read_csv('ques_model/test.csv')
    data = f['embed'].tolist()
    y = f['label_id'].tolist()
    # mask = f['question_type'].tolist()
    for i in tqdm(data):
        pre.predict(topK=5, examTXT=i, isVote=True)

    # y = [1]
    # ans = pre.predict(topK=5, examTXT='商店售货员小海把“收入100元”记作“＋100元”，那么“﹣60元”表示')
    # print(ans)

    acc = pre.topKacc(topk=20, depth=3, y=y)
    print(acc)
