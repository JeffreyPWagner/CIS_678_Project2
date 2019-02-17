import collections
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

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
docCou1nt = {}

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

# label for training classes
label = ""

testAnswers = []
testGuesses = []

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
    wordCount = Counter(docCombined[doc])
    docMinProb[doc] = 1 / docWordCount[doc]
    for w in wordCount:
        docWordProb[doc][w] = (wordCount[w] + 1) / docWordCount[doc]


# classifies a new document given a list of its words
def classifyDoc(wordList):

    # bring in global variables used
    global docTypes
    global docWordProb
    global docMinProb
    # set starting maximum probability to negative infinity
    maxProb = float('-inf')
    # find the most likely class for the given word list
    guessClass = ""
    for d in docTypes:
        # add the logs of the probabilities instead of multiplying the base figures to avoid underflow
        prob = math.log(docProbabilities[d])
        for word in wordList:
            if word in docWordProb[d].keys():
                prob += math.log(docWordProb[d][word])
            else:
                prob += math.log(docMinProb[d])
        # no need to convert the log figures back since they are only compared to each other
        if prob > maxProb:
            maxProb = prob
            guessClass = d
    return guessClass


def naiveBayes(exclusionRate):
    global allWords
    global vocab
    global testGuesses
    global testAnswers

    testAnswers.clear()
    testGuesses.clear()
    testFile = open(r"C:\Users\jeffp\OneDrive\Documents\GitHub\CIS_678_Project2\forumTest.data", "r")

    correctCount = 0
    mostCommon = [word for word, word_count in Counter(allWords).most_common(int(len(vocab) * exclusionRate))]

    for line in testFile:
        words1 = line.split()
        testAnswers.append(words1[0])
        del words1[0]
        wordsNoCommon = [word for word in words1 if word not in mostCommon]
        wordsNoShort = [word for word in wordsNoCommon if len(word) > 1]
        testGuesses.append(classifyDoc(wordsNoShort))

    testFile.close()

    for i, val in enumerate(testAnswers):
        if testGuesses[i] == val:
            correctCount += 1
    return correctCount / len(testGuesses)


def findOptimalExclusion():
    exclusionRate = 0.001
    maxAccuracy = -1
    bestExclusion = exclusionRate
    while exclusionRate < 0.011:
        accuracy = naiveBayes(exclusionRate)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestExclusion = exclusionRate
        exclusionRate += 0.001
    print('optimal exclusion: %r' % "{:.2%}".format(bestExclusion))
    return bestExclusion


naiveBayes(findOptimalExclusion())

docTypesListLong = list(docTypes)
docTypesList = [word[0:3] for word in docTypesListLong]

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# next 3 lines taken from https://stats.stackexchange.com/questions/51296
cm = confusion_matrix(testAnswers, testGuesses, labels=docTypesListLong)
recall = np.diag(cm) / np.sum(cm, axis=1)
precision = np.diag(cm) / np.sum(cm, axis=0)
print('recall: %r' % "{:.2%}".format(np.mean(recall)))
print('precision: %r' % "{:.2%}".format(np.mean(precision)))
cmDataFrame = pd.DataFrame(cm, index=docTypesList, columns=docTypesList)
print(cmDataFrame)


# TODO write resulting lists to file
# TODO potentially plot exclusion rates vs accuracy
# TODO add other accuracy metrics
# TODO clean up and remove unused

