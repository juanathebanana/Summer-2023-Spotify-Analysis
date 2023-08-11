import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
from langdetect import detect


df=pd.read_csv('spotify_data1.csv')
#print(df.columns)
df1=df


#here i can clean the data
def clean_lyrics(lyrics):
    # Check if lyrics is a string
    if isinstance(lyrics, str):
        # Lowercase the text
        lyrics = lyrics.lower()
        #delete character that won't be use for the analysis
        lyrics = re.sub(r'<.*?>', '', lyrics)

        lyrics = re.sub(r'[^\w\s]', '', lyrics)
        lyrics = re.sub(r'\d+', '', lyrics)

        try:
            language = detect(lyrics)
        except:
            language = 'english'

        if language == 'es':  # Spanish
            stop_words = set(stopwords.words('spanish'))
            stemmer = SnowballStemmer('spanish')
        elif language == 'pt':  # Portuguese
            stop_words = set(stopwords.words('portuguese'))
            stemmer = SnowballStemmer('portuguese')
        elif language == 'en':  # English
            stop_words = set(stopwords.words('english'))
            stemmer = SnowballStemmer('english')
        else:  # Other languages
            # Skip processing for other languages
            return lyrics

        words = word_tokenize(lyrics)
        lyrics = ' '.join([word for word in words if word.lower() not in stop_words])


        words = word_tokenize(lyrics)
        lyrics = ' '.join([stemmer.stem(word) for word in words])

        return lyrics
    else:
        return ''

#I create a new column in which I have all the lyrics cleaned
df1['clean_lyrics'] = df1['lyrics'].apply(clean_lyrics)




def toword(df):
    def unique(list1):
        # intilize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    # Stores unique words of each lyrics song into a new column called words
    words = []
    # iterate trought each lyric and split unique words appending the result into the words list
    df = df.reset_index(drop=True)
    for word in df['clean_lyrics'].tolist():
        words.append(unique(clean_lyrics(word).split()))
    # create the new column with the information of words lists
    df['words'] = words
    return df


df1=toword(df)
#create a new column in which is specified the language pf the song
from langdetect import detect

def detect_language(lyrics):
    try:
        # Detect the language
        language = detect(lyrics)
    except:
        # If language detection fails, set language as 'unknown'
        language = 'unknown'
    return language

df1['lyrics'] = df1['lyrics'].apply(lambda x: x if isinstance(x, str) else '')

df1['language'] = df1['lyrics'].apply(detect_language)
#print(df1['language'])

def remove_articles_and_conjunctions(words):
    # A list of common Portuguese articles and conjunctions to be removed
    stopwords_pt = ['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'e', 'ou', 'mas', 'porque', 'que', 'como', 'quando', 'onde', 'quem', 'de', 'do', 'da', 'dos', 'das', 'dum', 'duma', 'duns', 'dumas', 'em', 'no', 'na', 'nos', 'nas', 'num', 'numa', 'nuns', 'numas', 'por', 'pelo', 'pela', 'pelos', 'pelas', 'para', 'ao', 'à', 'aos', 'às', 'com', 'contra', 'entre']

    # Filter out the stopwords from the list of words
    filtered_words = [word for word in words if word.lower() not in stopwords_pt]

    return filtered_words

df1['words'] = df1['words'].apply(remove_articles_and_conjunctions)

