import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
from data_cleaning import df1

#just in english and spanish

df_en= df1[df1['language'] == 'en']
df_es=df1[df1['language'] == 'es']

# Generate a single string of all words in the 'words' column of df_es
all_words_en = ' '.join([str(word) for word in df_en['words']])

# Generate the word cloud english words
wordcloud_en = WordCloud(width=800, height=400, background_color='black').generate(all_words_en)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_en, interpolation='bilinear')
plt.axis("off")
#plt.show()


all_words_es = ' '.join([str(word) for word in df_es['words']])

# Generate the word cloud
wordcloud_es = WordCloud(width=800, height=400, background_color='white').generate(all_words_es)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_es, interpolation='bilinear')
plt.axis("off")
plt.show()

