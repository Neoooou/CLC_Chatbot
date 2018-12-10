from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering
from classifier.preprocessing import preprocessing
from sklearn.metrics import silhouette_score
import clustering.utils

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import adjusted_rand_score
from clustering import utils



# for using pca to do dimension reduction,
# find and plot the relationship between the dimension that reduce to and silhouette score
def tune_pca(X,k):
    d_range = range(2,100)
    scores = []
    for d in d_range:
        X_ = utils.perform_pca(X, d)
        model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, random_state=32)
        model.fit(X_)
        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(X_, cluster_labels)
        scores.append(silhouette_avg)
        print("PCA:{0}/{1}".format(d,silhouette_avg))
    plt.plot(d_range,scores)
    plt.title("Principle Component Analysis")
    plt.xlabel("Dimensionality")
    plt.ylabel("Silhouette Score")
    plt.show()
#for using locally linear embedding, find the optimized parameters
def tune_lle(X,k):
    d_range = range(2, 100)
    scores = []
    for d in d_range:
        X_ = utils.perform_lle(X,d)
        model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, random_state=32)
        model.fit(X_)
        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(X_, cluster_labels)
        scores.append(silhouette_avg)
        print("LLE:{0}/{1}".format(d, silhouette_avg))
    plt.plot(d_range, scores)
    plt.title("Locally Linear Embedding")
    plt.xlabel("Dimensionality")
    plt.ylabel("Silhouette Score")
    plt.show()
def tune_se(X,k):
    d_range = range(2, 100)
    scores = []
    for d in d_range:
        X_ = utils.perform_se(X, d)
        model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, random_state=32)
        model.fit(X_)
        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(X_, cluster_labels)
        scores.append(silhouette_avg)
        print("SE:{0}/{1}".format(d,silhouette_avg))
    plt.plot(d_range, scores)
    plt.title("Spectral Embedding")
    plt.xlabel("Dimensionality")
    plt.ylabel("Silhouette Score")
    plt.show()
# tune k-means parameters
def tune_kmeans(X):
    ks = range(2,100)
    # best dimension reduction method and optimized parameters we got
    #X_ = utils.perform_lle(X,3)
    scores = []
    for k in ks:
        model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, random_state=32)
        model.fit(X)
        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        scores.append(silhouette_avg)
        print("[LLE is applied]while number of clusters is",k,"silhouette average score is",silhouette_avg)
    plt.plot(ks, scores)
    plt.title("K-means")
    plt.xlabel("Predefined Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()
def scatter_samples(X_trn_,y_trn_,tittle="Spectral Clustering"):
    colors = cm.rainbow(np.linspace(0, 1, 10))
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection="3d")
    for i in range(len(X_trn_)):
        plt.scatter(X_trn_[i,0],X_trn_[i,1],c=colors[y_trn_[i]])
    plt.title(tittle)
#plt.xlim((-0.25,0.05))
#plt.xticks(np.linspace(-0,25,0.05,50))
pt = preprocessing("../crawler/clcjsondata.txt")
docs,labels,y = pt.process()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)
tune_pca(X,10)
tune_lle(X,10)
tune_se(X,10)
#tune_kmeans(X)
#X = utils.perform_lle(X,2)
#X_trn,X_tst,y_trn,y_tst = train_test_split(X,y,test_size=0.2)
#model = KMeans(n_clusters=10, init="k-means++", max_iter=100, n_init=1, random_state=32)
#model = AgglomerativeClustering(
#        n_clusters=10, linkage="average", affinity="euclidean")
#model = SpectralClustering(
#        n_clusters=10, eigen_solver='arpack', affinity='nearest_neighbors')
#model.fit(X_trn)
#labels = model.labels_
#y_pred = model.predict(X_tst)
#scatter_samples(X_trn,labels)
#labels_true = [0,0,1,1]
#labels_pred = [1,1,2,2]

# calculate a similarity measure between two clusterings by considering all pairs of samples
#and counting pairs of samples that are assigned to same or different clusters in the predicted and
# true clusterings, it is a simple accuracy
#ari = adjusted_rand_score(y_pred,y_tst)
#ri = utils.random_index(y_pred, y_tst)
#print("adjust random index score:",ari)
#plt.show()






















'''
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# print
# all tokens
terms = vectorizer.get_feature_names()
print('term:',terms)

# print all tokens that belongs to two clusters respectively
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["examples about breaking law"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["some was bullied at school"])
prediction = model.predict(Y)

print(prediction)'''