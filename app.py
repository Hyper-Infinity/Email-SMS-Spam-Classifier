import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB


cv = pickle.load(open('CountVetorizer.pkl', 'rb'))
bn = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    ans = list()
    for i in text:
        if i.isalnum():
            ans.append(i)

    text = ans[:]
    ans.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            ans.append(i)

    text = ans[:]
    ans.clear()

    for i in text:
        ans.append(ps.stem(i))

    return " ".join(ans)


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area('Enter the message, you want to check :', 'Hi there !')

if st.button("Predict"):
    # Text preprocessing
    trans_sms = transform_text(input_sms)

    # vectorization
    final_sms = cv.transform([trans_sms])

    # prediction
    result = bn.predict(final_sms)[0]

    # Display
    if result == 1:
        st.header('Sapm')
    else:
        st.header('Not Spam')


