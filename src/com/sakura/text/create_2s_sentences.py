import pandas as pd
import os
# import numpy as np
import uuid

PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
INPUT_DIR = PROJ_DIR + '/original_data/annotation/annotations.tsv'
OUTPUT_DIR = PROJ_DIR + '/output/annotation/annotations.tsv'

# 创建一个空的dataframe
df = pd.DataFrame(columns=["UUID", "Sentences", "Word Onset", "Word Offset", "token", "subword"])
train_df = pd.read_csv(INPUT_DIR, sep='\t', header=0)
# print(train_df)

for i in range(len(train_df)):
    j = len(df) - 1
    # add into the same line
    if j >= 0 and df['Word Onset'][j] <= train_df['Word Onset'][i] <= df['Word Onset'][j] + 2:
        # if train_df[][]
        df.loc[j, 'Sentences'] = df['Sentences'][j] + ' ' + train_df['Words'][i]
        # df.loc[j, 'subword'] = train_df['Words'][i]
        df.loc[j, 'Word Offset'] = train_df['Word Offset'][i]
        # print(train_df['Words'][i])
    # 标点符号。实际上不占时间
    elif j >= 0 and pd.isna(train_df['Word Onset'][i]):
        df.loc[j, 'Sentences'] = df['Sentences'][j] + ' '+ train_df['Words'][i]
        df.loc[j, 'subword'] = df['Sentences'][j] + train_df['Words'][i]
        # df.loc[j, 'Word Onset'] = train_df['Word Onset'][i - 1] + 2

    # the first line
    # create new line
    else:
        # word = train_df['Words'][i]
        if j < 0:
            onset = train_df['Word Onset'][j + 1]
        else:
            onset = df['Word Onset'][j] + 2
            # i = i-1
            df.loc[j, 'Word Offset'] = df['Word Onset'][j] + 2
            df.loc[j, 'subword'] = df['subword'][j] + ' ' + train_df['Words'][i]

        if i > 0 and train_df['Word Offset'][i - 1] > onset:
            word = train_df['Words'][i - 1] + ' ' + train_df['Words'][i]
        else:
            word = train_df['Words'][i]

        dict_data = {'UUID': uuid.uuid1(), 'Sentences': word,
                     'Word Onset': onset,
                     'Word Offset': onset + 2,
                     "token": '', "subword": train_df['Words'][i]}
        df = df.append(pd.DataFrame(dict_data, index=[len(df)]))
        continue
df.to_csv(OUTPUT_DIR, sep='\t')
# print(df)
