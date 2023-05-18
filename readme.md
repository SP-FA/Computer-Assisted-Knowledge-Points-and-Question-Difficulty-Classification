# Computer-Assisted Knowledge Points and Question Difficulty Classification Based on BERT

##### Abstract

>In recent years, with the increasing use of educational technology and online learning platforms, there has been a growing interest in developing intelligent systems that can automatically predict the knowledge points associated with educational questions. This paper presents a novel approach for knowledge point prediction in middle school mathematics questions. The dataset used in this study consists of a large collection of 591,379 middle school mathematics questions. To leverage the power of natural language processing techniques, the questions are preprocessed using a tokenizer and encoded into word embeddings using BERT (Bidirectional Encoder Representations from Transformers). Matrix dot production is then employed to calculate the similarity between each test question and the training set, and the top-n most similar vectors are selected for each test question. A voting mechanism is introduced to eliminate duplicate knowledge points and rank them based on their total scores in descending order, resulting in k candidate solutions. The performance of the proposed approach is evaluated using top-k accuracy as the evaluation metric and a lowest common ancestor (LCA) algorithm is used to calculate accuracy at different levels of the knowledge point tree. The results demonstrate the effectiveness of the proposed approach in predicting knowledge points for middle school mathematics questions and its potential for application in intelligent tutoring systems and educational assessment tools.
