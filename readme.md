# Computer-Assisted Knowledge Points and Question Difficulty Classification Based on BERT

![https://img.shields.io/badge/python-3.7-green](https://img.shields.io/badge/Python-3.7-green)
![https://img.shields.io/badge/pytorch-1.13.0-yellow](https://img.shields.io/badge/Pytorch-1.13.0-yellow)
![https://img.shields.io/badge/CUDA-12.1-brightgreen](https://img.shields.io/badge/CUDA-12.1-brightgreen)

##### Abstract

>In recent years, with the increasing use of educational technology and online learning platforms, there has been a growing interest in developing intelligent systems that can automatically predict the knowledge points associated with educational questions. This paper presents a novel approach for knowledge point prediction in middle school mathematics questions. The dataset used in this study consists of a large collection of 591,379 middle school mathematics questions. To leverage the power of natural language processing techniques, the questions are preprocessed using a tokenizer and encoded into word embeddings using BERT. Matrix dot production is then employed to calculate the similarity between each test question and the training set, and the top-n most similar vectors are selected for each test question. A voting mechanism is introduced to eliminate duplicate knowledge points and rank them based on their total scores in descending order, resulting in k candidate solutions. The performance of the proposed approach is evaluated using top-k accuracy as the evaluation metric and a LCA algorithm is used to calculate accuracy at different levels of the knowledge point tree. The results demonstrate the effectiveness of the proposed approach in predicting knowledge points for middle school mathematics questions and its potential for application in intelligent tutoring systems and educational assessment tools.

## Overview

This implements knowledge point type prediction using dot similarity and voting mechanism, and knowledge point difficulty prediction using BERT [1]. It also uses LCA [2] for multi-level accuracy evaluation.

The network structure of difficulty classification is shown as follows:

![./sources/pd.png](./sources/pd.png#pic_center)

## Get started

### Installation

Check the `requirements.txt`:

`pip install -r requirements.txt`

### Data

You need to prepare the following files under the `./data` folder:

1. Three `.csv` files are needed for test, train, and valid. Each file should contain the following columns:
    - exam_txt: Question stem
    - point_type: Knowledge point type
    - question_type: Question type
    - label_id: ID corresponding to knowledge point type
    - difficulty: Difficulty of the question
    - embed: Word embeddings after embedding
2. `category.json`: A multi-level knowledge point tree
3. `labelID.json`: The dictionary that corresponds knowledge points with IDs can be generated using the `getID` method in `./preQuestionPoint/GetID.py`
4. `questionType.json`: The dictionary that corresponds question types with IDs, for example:
```json
{
    "填空题": 0,
    "单选题": 1,
    "解答题": 2,
    "判断题": 3,
    "多选题": 4
}
```

## Reference

[1] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

[2] [Fast Algorithms for Finding Nearest Common Ancestors](https://epubs.siam.org/doi/10.1137/0213024)
