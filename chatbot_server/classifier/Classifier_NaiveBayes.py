from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.manifold import locally_linear_embedding
import sklearn
# get data
pt = preprocessing("../crawler/clcjsondata.txt")
docs,y, links = pt.process()
# convert paragraphs into vectors
v = TfidfVectorizer(stop_words="english")
X = v.fit_transform(docs).toarray()

# use RandomTreesEmbedding to transform data
#hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
#X_transformed = hasher.fit_transform(X)

# Visualize result after dimensionality reduction using truncated SVD
#svd = TruncatedSVD(n_components=2)
#X_reduced = svd.fit_transform(X_transformed)
#print(X_reduced)
#X_reduced,err = locally_linear_embedding(X,n_neighbors=12,n_components=3,random_state=32)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2,random_state=95)
#clf = BernoulliNB()
clf = sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
clf.fit(X_trn,y_trn)
y_pred = clf.predict(X_tst)
bool_result = np.equal(y_tst,y_pred)
true_result = np.sum(bool_result)
print("accuracy: %2.3f" % (true_result/len(bool_result) * 100))
