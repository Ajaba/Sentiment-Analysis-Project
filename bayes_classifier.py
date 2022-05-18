# My imports
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import regex
import pickle

# Import the Dataset, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist
dataSet = pd.read_csv("datasets/train_one.csv")
trainingDataSet = dataSet.tweet[:400]
trainingDataSetClassification = dataSet.label[:400]

testSet = pd.read_csv("datasets/train_one.csv")
testingDataSet = testSet.tweet[401:500]
testingDataSetClassification = testSet.label[401:500]

# Clean the data
tokenizer = RegexpTokenizer(r"\w+")
englishStopWords = set(stopwords.words("english"))

ps = PorterStemmer()

def cleanTextData(textInput):
    textInput = textInput.lower()
    
    # Tokenize
    # tokens = tokenizer.tokenize(textInput)
    tokens = word_tokenize(textInput) #
    new_tokens = []
    for w in tokens:
        resultText = regex.sub(r"[^\w\s]", "", w)
        resultText = resultText.replace("ð", "")
        resultText = resultText.replace("ï", "")
        resultText = resultText.replace("â", "")
        resultText = resultText.replace("ä", "")
        if resultText != "":
            new_tokens.append(resultText)

    listTokens = [token for token in new_tokens if not token in englishStopWords]

    # Stem the words
    stemmedTokens = [ps.stem(token) for token in listTokens]

    # Lemmitization
    lemWords = []
    wnet = WordNetLemmatizer()
    for w in stemmedTokens:
        lemWords.append(wnet.lemmatize(w))

    cleanedText = " ".join(lemWords)
    cleanedText = cleanedText.strip()

    if cleanedText != "":
        return cleanedText

cleanedTrainData = [cleanTextData(i) for i in trainingDataSet]
cleanedTestData = [cleanTextData(i) for i in testingDataSet]
# print(cleanedTestData)

# Vectorize Text
cv = CountVectorizer(ngram_range = (1, 2))
trainingDataVector = cv.fit_transform(cleanedTrainData).toarray()
testingDataVector = cv.transform(cleanedTestData).toarray()

# Text classification using Multinomial Naive Bayes (MNB)
# MNB is a type of Naive Bayes, this Naive Bayes is used in text classification
mN = MultinomialNB()
mN.fit(trainingDataVector, trainingDataSetClassification)
score = mN.score(testingDataVector, testingDataSetClassification)
print("Accuracy of model = " + str(score * 100) + "%\n")

# Saving the dataset
pickle.dump(cv, open("sentiment_model_vec.pkl","wb"))
pickle.dump(mN, open("sentiment_model.pkl","wb"))

# To import model for use
# cv = pickle.load(cv, open("sentiment_model_vec.pkl","rb"))
# mN = pickle.load(mN, open("sentiment_model.pkl","rb"))

while True:
    inputText = input("Enter sample text: ")
    inferText = [cleanTextData(inputText)]
    inferTextVector = cv.transform(inferText).toarray()
    inferTextResult = mN.predict(inferTextVector)
    if(inferTextResult[0] == 0):
        inferTextResult = "[POSTIVE]"
    else:
        inferTextResult = "[NEGATIVE]"
    print("'" + inputText + "' is " + inferTextResult + "\n")