
import scipy.stats as stats

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

models = [LinearSVC(random_state=0), MultinomialNB(), LogisticRegression(random_state=0)]
FOLDS = 5

def perform_cross_val(models, X_train, y_train, scoring, cv=FOLDS):
    """
    perform cross-validation model fitting
    and returns the results

    :param models: list of sci-kit learn model classes
    :param X_train: training data set
    :param y_train: response labels
    :param cv: # of folds
    :return: a df with the accuracies for each model and fold
    """
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

    return pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


def plot_cv_results(cv_df):
    """
    Plots the distributions of the accuracy metrics
    for each fold in each of our models.

    :param cv_df: the output df from perform_cross_val
    :return: None
    """
    plt.figure(figsize=[10, 10])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    f = sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    f.xaxis.tick_top() # x labels on top
    plt.setp(f.get_xticklabels(), rotation=30, fontsize=20) # rotate and increase x labels
    plt.show()
    # calculate accuracy mean & std
    display(cv_df.groupby('model_name')\
                 .agg({"accuracy": [np.mean, stats.sem]})\
                 .sort_values(by=('accuracy', 'mean'), ascending=False)\
                 .round(4))




X_train, X_dev_test, y_train, y_dev_test = train_test_split(train_features,
                                                            train_labels,\
                                                            test_size=.2,\
                                                            random_state = 0)

lb = preprocessing.LabelBinarizer()
lb.fit(y_train)

cv_scores = cross_val_score(models[0], X_train, y_train, scoring, cv=5)



# run cv function
results = perform_cross_val(models, X_train, y_train)

plot_cv_results(results)
