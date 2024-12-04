# sentiment analysis of Campus Wall posts
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

posts_data = pd.read_csv('posts_emotion.csv')


# 1 代表积极，0 代表消极
# 1 indicates positive，0 indicates negative
examples = [1, 0, 1, 0, 1] # just to take an example, as we need different data sets

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(posts_data['post_content'])
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# prediction
y_pred = clf.predict(X_test)

# accuracy
print(f"情感分析准确率: {accuracy_score(y_test, y_pred)}")

# predicting the sentiment of new posts
new_post = ["Had an amazing time at the event, it was so much fun!"]
new_post_vectorized = vectorizer.transform(new_post)
print(f"预测情感：{'积极' if clf.predict(new_post_vectorized) == 1 else '消极'}")
