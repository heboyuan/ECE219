from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import random
import nltk

# Question 1
twenty_train = fetch_20newsgroups(subset='train', shuffle = True, random_state = 42)

#doc_number = list(map(lambda category: len(fetch_20newsgroups(
#    subset='train', categories=[category]).data), twenty_train.target_names))
ind, counts = np.unique(twenty_train.target, return_counts=True)

bar_colors=[]
for i in range(len(counts)):
    rgb = np.random.rand(3,)
    bar_colors.append(rgb)

p1 = plt.bar(ind, counts, width = 0.8, color=bar_colors)
plt.ylabel('Number of documents')
plt.xlabel('Categories')
plt.title('Number of training docs in each category')
plt.xticks(ind, np.arange(1, len(counts)+1))
plt.legend(p1, twenty_train.target_names, bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 9})
plt.savefig("q1_G.png", bbox_inches="tight")

# Question 2
np.random.seed(42)
random.seed(42)

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = None)

import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
wnl = nltk.wordnet.WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent(list_word):
    # Text input is list of strings, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]

# overwrite analyzer with callable function:
from sklearn.feature_extraction.text import CountVectorizer

vectorizer1 = CountVectorizer(stop_words='english')
stop_words = vectorizer1.get_stop_words()
#print(stop_words)
print("Number of stop words are:")
print(len(stop_words))

analyzer = CountVectorizer().build_analyzer()

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in stop_words and not word.isdigit())
    #Stopwords and checking for digits must be done here itself.
    #Because, when a callable is passed to analyzer in CoutVectorizer, it extracts features from THE RAW, UNPROCESSED INPUT.

count_vect = CountVectorizer(min_df=3 ,analyzer=stem_rmv_punc) 
#G - I do not think stop_words = 'english' is necessary here because of the comments mentioned in stem_rmv_punc
X_train_counts = count_vect.fit_transform(train_dataset.data)
print("Shape of X_train_counts is")
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("Shape of X_train_tfidf is")
print(X_train_tfidf.shape)

