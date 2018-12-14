import clustering.utils as u
from classifier.preprocessing import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings
from sklearn.preprocessing import OneHotEncoder
def svm_trn_pred(X,y,c,g):
    # divide data into training dara and testing data
    X_trn, X_test, y_trn, y_test = train_test_split(X, y, test_size=0.2,random_state=95)
    # SVC impalement the "one-vs-one" approach for multi-class classification
    clf = SVC(kernel="rbf",C=c,gamma=g,random_state=95)
    clf.fit(X_trn,y_trn)
    y_pred = clf.predict(X_test)
    correct_pred = 0
    leny = len(y_pred)
    for i in range(leny):
        if y_test[i] == y_pred[i]:
            correct_pred += 1
    acc = correct_pred / leny * 100
    print("while c = ",c," and gamma = ",g,"accuarcy = %2.3f" % acc,"%")
def tune_parameters(X,y,tuned_parameters):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scores = ['precision', 'recall']
    warnings.filterwarnings('ignore')
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
# get data
pt = preprocessing("../crawler/clcjsondata.txt")
docs,y, links = pt.process()
# convert paragraphs into vectors
X = u.tfidf_vectorize(docs)

# encode labels using one-hot encoder
# y = onthot_encoder(y.reshape(-1,1))  ! error: y width should be 1 !!!

# enc = OneHotEncoder()
# X = enc.fit_transform(X.toarray())  ! doesn't work !!!

# perform pca to reduce input's dimension
# X = perform_pca(X,300)        ! improve the compute speed, but no performance improved !!!

# use GridSearchCV to tune parameters
tuned_parameters = [{'kernel': ['sigmoid','linear','poly','rbf'], 'gamma': [1e-1,1e-2,1e-3],
                     'C': [1, 10, 30, 50,100]}
                    ]
tune_parameters(X,y,tuned_parameters)
#svm_trn_pred(X,y,5.0,0.1)
#tfidf_vectorizer(docs1)




