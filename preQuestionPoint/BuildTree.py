import json
import matplotlib.pyplot as plt
import seaborn as sns


TreeNodes = {}

class TreeNode:
    def __init__(self, label, depth, father=None, children=None):
        self.label = label # String
        self.depth = depth # int
        self.father = self if father == None else father # TreeNode
        self.children = [] if children == None else children # List


def buildTree(root, lst, depth):
    TreeNodes[root.label] = root
    if (lst == None):
        return
    childLst = []
    for i in lst:
        node = TreeNode(i["label"], depth, root)
        childLst.append(node)
        buildTree(node, i["children"], depth+1)
    root.children = childLst


def printTree(node):
    print(node.label)
    print(node.children)
    for i in node.children:
        printTree(i)


def LCA(a, b):
    if (a.depth < b.depth):
        a, b = b, a
    for i in range(a.depth - b.depth):
        a = a.father
    while a != b:
        a = a.father
        b = b.father

    return a


if __name__ == "__main__":
    dct = json.load(open("ques_model/category.json", 'r', encoding='utf-8'))
    root = TreeNode("root", 0)
    buildTree(root, dct, 1)
    a = TreeNodes["数与式"]
    b = TreeNodes["正负数的实际应用"]
    lca = LCA(a, b)
    print(lca.label)
    coutn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in TreeNodes.values():
        coutn[i.depth] += 1

    print(coutn)
    sns.lineplot(data=coutn[1:7])
    plt.savefig("tree.png")

