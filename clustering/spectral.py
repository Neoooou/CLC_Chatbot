from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
pt = preprocessing("../crawler/clcjsondata.txt")
docs,labels,y = pt.process()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs).toarray()
X_trn,X_tst,y_trn,y_tst = train_test_split(X,y,test_size=0.2)
spectral = SpectralClustering(
        n_clusters=10, eigen_solver='arpack',
        affinity="nearest_neighbors")
y_trn_pred=spectral.fit_predict(X_trn)
ari = adjusted_rand_score(y_trn_pred,y_trn)
print(ari)

