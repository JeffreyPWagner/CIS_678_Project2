import collections
import math
from collections import Counter

# set of all words in docs
vocab = set()

# list of all words
allWords = []

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

# dictionary of dictionaries of word probabilities for each doc type
docWordProb = collections.defaultdict(dict)

# dictionary of minimum word probabilities for each doc type
docMinProb = {}

# total number of documents
numDocs = 0

# list of test doc types
testTypes = []

# list of assigned classes
testGuesses = []

# label for training classes
label = ""

# count of correct class assignments
correctCount = 0

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
    allWords += words
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
    docMinProb[doc] = 1 / docWordCount[doc]
    for w in wordcount:
        docWordProb[doc][w] = (wordcount[w] + 1) / docWordCount[doc]

mostCommon = [word for word, word_count in Counter(allWords).most_common(int(len(vocab) * .002))]

testFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\CIS_678_Project2\forumTest.data", "r")

for row in testFile:
    words = row.split()
    testTypes.append(words[0])
    del words[0]
    wordsMod = [word for word in words if word not in mostCommon]
    maxProb = -999999999
    # TODO set to min
    guessClass = label
    for doc in docTypes:
        prob = math.log(docProbabilities[doc])
        for word in wordsMod:
            if word in docWordProb[doc].keys():
                prob += math.log(docWordProb[doc][word])
            else:
                prob += math.log(docMinProb[doc])
        if prob > maxProb:
            maxProb = prob
            guessClass = doc
    testGuesses.append(guessClass)

for i, val in enumerate(testTypes):
    if testGuesses[i] == val:
        correctCount += 1

print(correctCount / len(testGuesses))
# TODO email prof to see if additional test sets are available to test for overtraining
# TODO try additional steps in assignment
