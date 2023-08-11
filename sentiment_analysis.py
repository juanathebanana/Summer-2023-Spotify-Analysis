import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from puli import df1


nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

comp_score = []
sentiment = []


df_en = df1[df1['language'] == 'en']

df_en = df_en[['artist', 'name', 'clean_lyrics','words']]

def remove_stopwords(text):  # text is a list/series of string to clean
    clean_text = list()
    nltk.download('punkt')
    nltk.download('stopwords')
#common slang words
    words = ["yeah", "ya", "na", "wan", "uh", "gon", "ima", "mm", "uhhuh", "bout", "em", "nigga", "niggas", "got", "ta",
             "lil", "ol", "hey",
             "oooh", "ooh", "oh", "youre", "dont", "im", "youve", "ive", "theres", "ill", "yaka", "lalalala", "la",
             "da", "di", "yuh",
             "shawty", "oohooh", "shoorah", "mmmmmm", "ook", "bidibambambambam", "shh", "bro", "ho", "aint", "cant",
             "know", "bambam",
             "shitll", "tonka","eas"]
    ignore = (stopwords.words('english') + words)

    for i in text:
        words = nltk.word_tokenize(i)
        # for i in range(len(words)):
        #    words = [w for w in words if w not in stopwords.words('english')]
        for element in ignore:  # given the tokenized list, return a list that doesn't contain any of the elements
            words = list(filter(lambda x: x != element and len(x) > 1, words))
        lyric = " ".join(words)
        clean_text.append(lyric)

    return clean_text


df_en['LyricsClean'] = remove_stopwords(df_en['clean_lyrics'])


sia = SentimentIntensityAnalyzer()

comp_score = []
sentiment = []

for i in df_en.loc[:,'LyricsClean']:
    sentiment.append(sia.polarity_scores(i))

# Compound score is the sum of positive, negative & neutral scores
# which is then normalized between -1(most extreme negative) and +1 (most extreme positive).
df_en.loc[:,'sent_scores'] = sentiment
df_en.loc[:,'comp_score'] = df_en.loc[:,'sent_scores'].apply(lambda x: x['compound'])
df_en.loc[:,'sentiment'] = df_en.loc[:,'comp_score'].apply(lambda x: 'Positive' if x>=0.5 else 'Negative' if x<=-0.5 else 'Neutral')

df_en.to_csv('sentiment.csv', index=False)


# Set up the matplotlib figure
plt.figure(figsize=(12, 6))

sns.countplot(x='sentiment', data=df_en, order=['Positive', 'Neutral', 'Negative'])
plt.title('Count of Each Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

