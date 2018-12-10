from classifier.preprocessing import preprocessing
from clustering import utils
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,SpectralClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score,v_measure_score
import matplotlib.pyplot as plt
if __name__ == "__main__":
    pp = preprocessing("../crawler/clcjsondata.txt")
    docs,labels,y = pp.process()
    # vectorize documents with various word embedding methods
    X_tfidf = utils.tfidf_vectorize(docs)
    X_tfidf = utils.perform_lle(X_tfidf,9)
    X_count = utils.count_vectorize(docs)
    X_count = utils.perform_lle(X_count,9)
    X_onehot = utils.onthot_encode(X_count)
    X_onehot = utils.perform_lle(X_onehot,9)
    vectorizers = (("TF-IDF",X_tfidf),("COUNT",X_count),("ONE-HOT",X_onehot))
    params = {"n_clusters":10,
              "eps":.5,
              "min_pts":5}
    # initialize clusters
    model_KM = KMeans(
        n_clusters=params["n_clusters"], init="k-means++", max_iter=100, n_init=1, random_state=32)
    model_DBSCAN_E = DBSCAN()
    model_DBSCAN_C = DBSCAN()
    model_hierarchical_E = AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="average", affinity="euclidean")
    model_hierarchical_C = AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="average", affinity="cosine")
    model_spectral = SpectralClustering(
        n_clusters=params["n_clusters"], eigen_solver='arpack', affinity='nearest_neighbors')
    clustering_algorithms = (("K-Means", model_KM),
                             ("HCE", model_hierarchical_E),#Hierarchical Clustering with Euclidean
                             #("Hierarchical Clustering with Cosine", model_hierarchical_C),
                             ("SC",model_spectral))#Spectral Clustering

    table_row = []
    row_label = [n for n,m in clustering_algorithms]
    print(row_label)
    col_label = ["silhouette score","Random Index","Adjusted Random Index","NMI"]
    for vectorizer_name,x in vectorizers:
        table_data = []
        fig,axs = plt.subplots(2,1)
        axs[0].axis('tight')
        axs[0].axis('off')
        for algorithm_name,model in clustering_algorithms:
            y_pred = model.fit_predict(x)
            silhouette_avg_score = round(silhouette_score(x,y_pred),2)
            ri = round(utils.random_index(y_pred,y),2)
            ari = round(adjusted_rand_score(y_pred,y),2)
            nmi = round(v_measure_score(y_pred,y),2)
            print(vectorizer_name," ",algorithm_name," silhouette average score=", silhouette_avg_score,
                  "ri=",ri," ari=",ari," nmi=",nmi)
            table_row = [silhouette_avg_score,ri,ari,nmi]
            table_data.append(table_row)
        axs[0].table(cellText=table_data,colLabels=col_label,loc='center',rowLabels=row_label)
        plt.title(vectorizer_name)
    plt.show()





