import pandas as pd

train = pd.read_csv('Dataset/train_en.tsv', delimiter='\t',encoding='utf-8')
dev = pd.read_csv('Dataset/dev_en.tsv', delimiter='\t',encoding='utf-8')
test = pd.read_csv('Dataset/test_en.tsv', delimiter='\t',encoding='utf-8')

print(train.shape, dev.shape, test.shape)
