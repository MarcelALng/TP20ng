# TP 20ng - Classification de textes
# Fait par : Marcel Albert Nguyen
# Établissement : Master NLP

# %% [markdown]
# ### Étape 1 : Les outils et la graine magique (Random Seed)
# Pour commencer, on a besoin de quelques outils. On va importer "scikit-learn" 
# qui est une bibliothèque super connue pour faire de l'intelligence artificielle simple.
# On fixe aussi une "random seed" pour que si on relance le programme,
# on ait exactement les mêmes résultats (c'est comme si on trichait avec le hasard).

# %%
import time # Pour calculer le temps que prend notre ordinateur
import numpy as np # Pour gérer les grands tableaux de chiffres
from sklearn.datasets import fetch_20newsgroups # Pour aller chercher les données
from sklearn.feature_extraction.text import TfidfVectorizer # Pour transformer les mots en nombres
from sklearn.naive_bayes import MultinomialNB # Notre modèle "cerveau" très simple
from sklearn.metrics import accuracy_score, classification_report # Pour voir si on a bien travaillé
from sklearn.model_selection import GridSearchCV # Pour chercher les réglages optimaux

# On définit la graine du hasard à 42 (la réponse à tout, selon certains !)
GRAINE_HASARD = 42
np.random.seed(GRAINE_HASARD)

print("C'est bon, on a tout ce qu'il faut !")
