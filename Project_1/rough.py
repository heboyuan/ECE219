# one example for advanced stemming
# The lemmatizer is actually pretty complicated, it needs Parts of Speech (POS) tags
import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
# nltk.download('punkt')#, if you need "tokenizers/punkt/english.pickle", choose it
# nltk.download('averaged_perceptron_tagger')
wnl = nltk.wordnet.WordNetLemmatizer()
#walking_tagged = pos_tag(nltk.word_tokenize('He is walking to school'))
#print(walking_tagged)
#print("mapping to Verb, Noun, Adjective, Adverbial")

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent_demo(text):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(nltk.word_tokenize(text))]

def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]

print(lemmatize_sent_demo('He is -23.45 9.8 walking to school'))
print("**************************")
# overwrite analyzer with callable function:
from sklearn.feature_extraction.text import CountVectorizer

vectorizer1 = CountVectorizer(min_df=3, stop_words='english')
stop_words = vectorizer1.get_stop_words()
#print(stop_words)
print(len(stop_words))

analyzer = CountVectorizer().build_analyzer()
print(analyzer('Hello, I am GG. h 5.98 89.eight 85.9 -23 -23!56.7'))
print(lemmatize_sent(analyzer('Hello, I am GG. h 5.98 89.eight 85.9 -23 -26.7')))
print("********************")
def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in stop_words and not word.isdigit())#  
    #Stopwords and checking for digits is done here itself.
    # Because, if a callable is passed to analyzer, it extracts features from THE RAW, UNPROCESSED INPUT.

cc = ['Hello, I am GG. h 5.98 89.eightttt 85.9 -23 -23!56.7', 'Above be eightttt  but Age & weight = -22 100 and -70.5']
count_vect1 = CountVectorizer(analyzer=stem_rmv_punc) 
count_vect = CountVectorizer(min_df = 1, analyzer=stem_rmv_punc) 
X_2 = count_vect1.fit_transform(cc)
print(count_vect1.get_feature_names())
