# ==========================================
# ==========================================
# ===    CODE FOR STACKING CLASSIFIER    ===
# ==========================================
# ==========================================
__date__ = '20230324'
__revised__='20230325'
__author__ = 'Jeremy Charlier'
P = print
#
#
import sklearn
print('sklearn.__version__:', sklearn.__version__)
from sklearn.datasets import load_breast_cancer
#
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBT
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline as mpip
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier as stackClf
# 
from sklearn.preprocessing import StandardScaler as sscl
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.inspection import permutation_importance
#
from itertools import combinations
#
#
# === STACKING FUNCTIONS ===
# get a list of models to evaluate
def get_models():
	models = dict()
	models['knn'] = mpip(
		sscl(), KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
	)
	models['bayes'] = mpip(sscl(), GaussianNB())
	models['svm'] = mpip(sscl(), SVC(C = 1.0, random_state = 0))
	models['lr'] = mpip(
		sscl(), LogisticRegression(C = 1.0, random_state = 0, n_jobs = -1))
	models['cart'] = DecisionTreeClassifier(max_depth = None)
	models['adaboost'] = AdaBoostClassifier(
		n_estimators = 50, learning_rate = 1.0, random_state = 0
	)
	models['gbt'] = GBT(
		learning_rate = 0.1, n_estimators = 100, random_state = 0
	)
	models['rf'] = RF(
		n_estimators = 100, max_depth = None, random_state = 0, n_jobs = -1
	)
	models['xgb'] = XGBClassifier(seed = 0)
	return models
#
def searchForBestStacking(stackList, X, y, NCV = 3):
  allClfs = get_models() # get individual models
  level1 = LogisticRegression() # define meta learner model
  results, names = list(), list()
  bestClf, bestScore = None, 0
  for iCombo in stackList:
    l0 = list()
    for iclf in iCombo:
      l0.append((iclf, allClfs[iclf])) # define level 0 estimators
    model = stackClf( # define the stacking ensemble
      estimators = l0,
      final_estimator = level1,
      cv = NCV
    )
    scores = evaluate_model(model, X, y)
    meanscrs = mean(scores)
    if meanscrs > bestScore:
      bestClf = iCombo
      bestScore = meanscrs
    results.append(scores)
    names.append(iCombo)
    print('> %s %.3f (%.3f)' % (iCombo, meanscrs, std(scores)))
    searchRes = Bunch(
      names = names, results = results,
      bestClf = bestClf, bestScore = bestScore
    )
  return searchRes
#
def getStacking(bestStackClf, NCV = 3):
  l0 = list()
  allClfs = get_models()
  level1 = LogisticRegression() # define meta learner model
  for iclf in bestStackClf:
    l0.append((iclf, allClfs[iclf])) # define level 0 estimators
  model = stackClf( # define the stacking ensemble
    estimators = l0,
    final_estimator = level1,
    cv = NCV
  )
  return model
#
# evaluate a given model using cross-validation
def evaluate_model(model, X, y, NSPLITS = 10, NREPEATS = 3):
	cv = RepeatedStratifiedKFold(
		n_splits = NSPLITS, n_repeats = NREPEATS, random_state = 1
	)
	scores = cross_val_score(
		model, X, y, scoring = 'average_precision', # roc_auc
		cv = cv, n_jobs = -1, error_score='raise'
	)
	return scores
#
def permutationImportance(model, xval, yval, NREPEATS = 30, THRESHOLD = 0.001):
  r = permutation_importance(
    model, xval, yval, n_repeats=NREPEATS, random_state=0
  )
  P('\n === PERMUTATION IMPORTANCE ===')
  for i in r.importances_mean.argsort()[::-1]:
    # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    if r.importances_mean[i] > THRESHOLD:
      P('> %-25s: %.3f +/- %.3f' % (
        xval.columns[i], r.importances_mean[i], r.importances_std[i])
      )
#
#
# === MAIN ===
# ============
#
# === DATA CODE ===
data = load_breast_cancer()
for item in range(len(data.feature_names)):
  data.feature_names[item] = data.feature_names[item].replace(' ', '_')
df = pd.DataFrame(data.data, columns = data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
  df,
  data.target,
  test_size = 0.3,
  shuffle = True
)
# 
# === EVALUATE MODELS ===
models = get_models()
results, names = list(), list()
P('\n=== INDIVIDUAL CLASSIFIER PERFORMANCE ===')
for name, model in models.items():
	scores = evaluate_model(model, X_train, y_train)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
#
# === STACK CODE ===
# ==================
NMINMDLS = 2 # min nbr of estimators in level 0
NMAXMDLS = 2 # max nbr of estimators in level 0
LISTMDLS = [ # same as dict keys in get_models
	'bayes', 'knn', 'svm', 'lr', 'cart',
  'adaboost', 'gbt', 'rf', 'xgb'
]
#
stackList = list()
for n in range(NMINMDLS, NMAXMDLS + 1):
	stackList.extend(list(combinations(LISTMDLS, n)))
P('\n=== SEARCH FOR BEST STACKING CLASSIFIER ===')
stackRes = searchForBestStacking(stackList, X_train, y_train)
bestStackClfList = stackRes.bestClf
P('\n> best stacking model:', bestStackClfList)
#
# train on xtrain and predict on xtest
bestStackClf = getStacking(bestStackClfList)
bestStackClf.fit(X_train, y_train)
yscores = bestStackClf.predict_proba(X_test)
#
# === FEATURES IMPORTANCE FOR STACKING CLASSIFIER === 
permutationImportance(bestStackClf, X_test, y_test)
