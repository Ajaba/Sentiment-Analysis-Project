# My imports
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Import the Dataset, where 1 is positive and 0 is negative
trainingDataSet = [
    "This was awesome, an awesome movie",
    "Great movie! I liked it a lot",
    "Happy ending! Awesome acting by the hero",
    "Loved it! Truely great",
    "Bad! Not up to the hype",
    "Could've been better",
    "Surely a disappoining movie"
]
trainingDataSetClassification = [1, 1, 1, 1, 0, 0, 0]

testingDataSet = [
    "I liked this movie!",
]

# Clean the data
tokenizer = RegexpTokenizer(r"\w+")
englishStopWords = set(stopwords.words("english"))

ps = PorterStemmer()

def cleanTextData(textInput):
    textInput = textInput.lower()
    
    # Tokenize
    tokens = tokenizer.tokenize(textInput)
    listTokens = [token for token in tokens if token not in englishStopWords]

    # Stem the words
    stemmedTokens = [ps.stem(token) for token in listTokens]

    cleanedText = " ".join(stemmedTokens)

    return cleanedText

cleanedTrainData = [cleanTextData(i) for i in trainingDataSet]
cleanedTestData = [cleanTextData(i) for i in testingDataSet]

# Vectorize Text
cv = CountVectorizer(ngram_range = (1, 2))
trainingDataVector = cv.fit_transform(cleanedTrainData).toarray()
testingDataVector = cv.fit_transform(cleanedTestData).toarray()

# Text classification using Multinomial Naive Bayes (MNB)
# MNB is a type of Naive Bayes, this Naive Bayes is used in text classification
mN = MultinomialNB()
mN.fit(trainingDataVector, trainingDataSetClassification)

testingDataResults = mN.predict(testingDataVector)

x = 0
for i in testingDataSet:
    print("[" + i + "] = " + str(testingDataResults[x]))
    x += 1