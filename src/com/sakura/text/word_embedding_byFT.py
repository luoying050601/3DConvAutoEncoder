from gensim.models.fasttext import FastText
import nltk
import pandas as pd
import os
import re
import time
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import wikipedia

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
DATA_DIR = PROJ_DIR + '/output/annotation/annotations.tsv'

en_stop = set(nltk.corpus.stopwords.words('english'))
# Creating Words Representation
embedding_size = 1024
window_size = 40
min_word = 5
down_sampling = 1e-2


# Data Preprocessing
def base_preprocess(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    return document.lower()


def preprocess_text(document):
    document = base_preprocess(document)
    # Lemmatization
    tokens = document.split()
    stemmer = WordNetLemmatizer()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    # preprocessed_text = ' '.join(tokens)

    return tokens


def main():
    # read from file
    # Scraping Wikipedia Articles
    # try:
    artificial_intelligence = wikipedia.page(title="Artificial Intelligence").content
    # machine_learning = wikipedia.page(title="Machine Learning").content
    deep_learning = wikipedia.page(title="Deep Learning").content
    neural_network = wikipedia.page(title="Neural Network").content
    artificial_intelligence = sent_tokenize(artificial_intelligence)
    # machine_learning = sent_tokenize(machine_learning)
    deep_learning = sent_tokenize(deep_learning)
    neural_network = sent_tokenize(neural_network)
    # artificial_intelligence.extend(machine_learning)
    artificial_intelligence.extend(deep_learning)
    artificial_intelligence.extend(neural_network)
    final_corpus = [' '.join(preprocess_text(sentence)) for sentence in artificial_intelligence if
                    sentence.strip() != '']

    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

    ft_model = FastText(word_tokenized_corpus,
                        size=embedding_size,
                        window=window_size,
                        min_count=min_word,
                        sample=down_sampling,
                        sg=1,
                        iter=100)
    df = pd.read_csv(DATA_DIR, sep='\t', header=0)
    for i in range(len(df)):
        sentences = df['Sentences'][i]
        sent_list = base_preprocess(sentences).split()
        for s in sent_list:
            # if s == df['subword'][i]:
            #     embedding = ft_model.wv[s]
            print(s)
            # if s == df['subword'][i]:
            #     s = ft_model.word_ngrams
            print(ft_model.wv[s])


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("time-cost:", str((end - start) / 60))
