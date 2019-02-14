import collections
from collections import Counter

# set of all words in docs
vocab = set()

# count of words in vocab
countVocab = 0

# list of all document labels
allDocs = []

# set of all document labels
docTypes = set()

# dictionary of count of each doc type
docCount = {}

# dictionary of probabilities of each doc type
docProbabilities = {}

# dictionary of concatenated docs of each type
docCombined = {}

# dictionary of word counts for each doc type plus count of words in vocab
docWordCount = {}

# dictionary of list of word probabilities for each doc type
docWordProb = collections.defaultdict(dict)

# total number of documents
numDocs = 0

# open and read training file
trainingFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\CIS_678_Project2\forumTraining.data", "r")

# read data from training file and close
for row in trainingFile:
    words = row.split()
    label = words[0]
    allDocs.append(label)
    docTypes.add(label)
    if label in docCombined.keys():
        docCombined[label] += words
    else:
        docCombined[label] = words
    del words[0]
    for w in words:
        vocab.add(w)
trainingFile.close()

# count total words in vocab
countVocab = len(vocab)

# count total number of docs and count number of each type
numDocs = len(allDocs)
docCount = Counter(allDocs)

# calculate probabilities, word counts, and word probabilities for each doc type
for doc in docTypes:
    docProbabilities[doc] = docCount[doc] / numDocs
    docWordCount[doc] = len(docCombined[doc]) + countVocab
    wordcount = Counter(docCombined[doc])
    for w in wordcount:
        docWordProb[doc][w] = (wordcount[w] + 1) / docWordCount[doc]
    # TODO create dictionary of minimum probabilities to reference in classifying step

print(docWordProb["atheism"])


