from sklearn import decomposition
from sklearn.manifold import locally_linear_embedding,SpectralEmbedding
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
def do1():
    print("do1...")
# calculate random index, a measure of the similarity between two data clusterings
def random_index(y_pred,y_true):
    if len(y_pred) != len(y_true):
        return None
    pairs_pred = get_all_pairs(y_pred)
    pairs_true = get_all_pairs(y_true)
    a,b,ri = 0,0,0.0
    # a, the number of pairs that are same in both y_pred and y_true
    # b, the number of pairs that are different in both y_pred and y_true
    # c, the number of pairs that are same in y_pred but different in y_true
    # d, the number of pairs that are same in y_true but different in y_pred
    # ri = (a+b)/(a+b+c+d)
    for p,t in zip(pairs_pred,pairs_true):
        if p[0] == p[1] and t[0] == t[1]:
            a += 1
        elif p[0] != p[1] and t[0] != t[1]:
            b += 1
    ri = round((a+b) / len(pairs_pred),2)
    return ri
def get_all_pairs(arr):
    if len(arr) < 1:
        return None
    pairs = []
    for i in range(len(arr) - 1):
        for j in range(i+1,len(arr)):
            pair = (arr[i],arr[j])
            pairs.append(pair)
    return pairs

#--------------Dimension Reduction Methods------
# perform Principle Component Analysis to reduce dimension
def perform_pca(X,i):
    X = X.todense()
    pca = decomposition.PCA(n_components=i, random_state=32)
    pca.fit(X)
    X = pca.transform(X)
    return X
# perform locally linear embedding(lle) to reduce dimension
def perform_lle(X,d):
    X_r,err = locally_linear_embedding(X.toarray(),n_neighbors=12,n_components=d,random_state=32)
    print("Done. Reconstruction error: %g"% err)
    return X_r
# perform Spectral Embedding to reduce dimension
def perform_se(X,d):
    X = X.todense()
    X_ = SpectralEmbedding(n_components=d, random_state=32).fit_transform(X)
    return X_

# -------------------text vectorizer methods---------------------

def count_vectorize(docs):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)
    return X

def tfidf_vectorize(docs):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)
    return X




def onthot_encode(matr):
    enc = OneHotEncoder()
    X = enc.fit_transform(matr)
    return X

if __name__ == "__main__":
    A = [1,2,0,3,2,2]
    B = [2,1,2,1,1,5]
    print(random_index(A,B))