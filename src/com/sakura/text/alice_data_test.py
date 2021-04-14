import os
import pandas as pd
import numpy as np

PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
annotation_file = PROJ_DIR + '/output/annotation/annotations.tsv'


def count_words(s):
    count = len(s.split())-1
    return count


alice_df = pd.read_csv(annotation_file, sep='\t', header=0)
alice_sent = alice_df['Sentences']
count_list = []
for i in range(len(alice_sent)):
    count_list.append(count_words(alice_sent[i]))
    print(count_list[i])
print(np.average(count_list))
