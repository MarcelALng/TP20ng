import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # TP 20ng - Classification de textes
    Marcel  Nguyen
    22500971
    19 décembre 2025
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    On fixe aussi une `random_seed` pour que si on relance le programme, on ait exactement les mêmes résultats.
    """)
    return


@app.cell
def _():
    import time
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import GridSearchCV

    random_seed = 42
    np.random.seed(random_seed)


    debut_total = time.time() #ajout pour trouver temps du calcul
    return (
        GridSearchCV,
        MultinomialNB,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        debut_total,
        fetch_20newsgroups,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    On va charger les données du jeu "20 Newsgroups" (des vieux messages de forums).
    On doit enlever les headers, footers et quotes comme.
    """)
    return


@app.cell
def _(fetch_20newsgroups):
    # On télécharge les données d'entraînement
    data_train = fetch_20newsgroups(
        subset="train", 
        remove=("headers", "footers", "quotes"),
        random_state=42 # On remet la graine ici aussi pour être sûr
    )

    # On télécharge les données de test 
    data_test = fetch_20newsgroups(
        subset="test", 
        remove=("headers", "footers", "quotes"),
        random_state=42
    )

    print(f"On a chargé {len(data_train.data)} messages pour l'entraînement.")
    print(f"On a chargé {len(data_test.data)} messages pour le test.")
    return data_test, data_train


@app.cell
def _(mo):
    mo.md(r"""
    On utilise `TfidfVectorizer`. C'est un outil qui compte les mots, mais qui donne moins d'importance aux mots trop fréquents (comme "the", "a") et plus d'importance aux mots rares qui veulent dire quelque chose.
    """)
    return


@app.cell
def _(TfidfVectorizer, data_test, data_train):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000) 
    # limite à 10000 pour accélérer le calcul

    X_train = vectorizer.fit_transform(data_train.data)
    y_train = data_train.target

    X_test = vectorizer.transform(data_test.data)
    y_test = data_test.target

    print("La transformation est finie !")
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    Entraînement et recherche du meilleur réglage
    On utilise un modèle appelé `MultinomialNB` (Naive Bayes). C'est un classique très efficace pour classer des textes.
    On va aussi utiliser `GridSearchCV` pour essayer plusieurs réglages (le paramètre `alpha`) et voir lequel est le meilleur.
    """)
    return


@app.cell
def _(GridSearchCV, MultinomialNB, X_train, y_train):
    # On définit les réglages à tester
    parametres = {'alpha': [0.01, 0.1, 1.0]}

    # On crée le chercheur de réglages
    modele_recherche = GridSearchCV(MultinomialNB(), parametres, cv=5)

    # On lance la recherche (ça peut prendre quelques secondes)
    modele_recherche.fit(X_train, y_train)

    # On récupère le meilleur modèle
    meilleur_modele = modele_recherche.best_estimator_

    print(f"Le meilleur réglage trouvé est : {modele_recherche.best_params_}")
    return (meilleur_modele,)


@app.cell
def _(mo):
    mo.md(r"""
    Évaluation des résultats
    Maintenant on regarde si notre modèle est bon en lui faisant prédire les catégories des messages de test qu'il n'a jamais vus.
    """)
    return


@app.cell
def _(
    X_test,
    accuracy_score,
    classification_report,
    data_test,
    meilleur_modele,
    y_test,
):
    # On fait des prédictions
    predictions = meilleur_modele.predict(X_test)

    # On calcule le score total
    score = accuracy_score(y_test, predictions)

    # On génère le rapport complet
    rapport = classification_report(y_test, predictions, target_names=data_test.target_names)

    print(f"Précision totale : {score:.2%}")
    return rapport, score


@app.cell
def _(mo):
    mo.md(r"""
    Conclusion et Temps d'exécution
    """)
    return


@app.cell
def _(debut_total, rapport, score, time):
    fin_total = time.time()
    temps_total = fin_total - debut_total

    # Un MacBook Pro M1 consomme environ 30 Watts (W) lorsqu'il travaille à plein régime.
    # C'est une estimation moyenne qui inclut le processeur M1 et la mémoire (16 Go).
    puissance_W = 30
    # On calcule l'énergie en Joules (J) : J = Watts * secondes
    conso_joules = puissance_W * temps_total

    print("=== RÉSULTATS FINAUX ===")
    print(f"Temps de calcul total : {temps_total:.2f} secondes")
    print(f"Puissance utilisée (estimée) : {puissance_W} W")
    print(f"Énergie consommée (estimée) : {conso_joules:.2f} Joules")
    print("-" * 30)
    print(f"Score final (Accuracy) : {score:.4f}")
    print("\nDétails par catégorie :")
    print(rapport)
    return


if __name__ == "__main__":
    app.run()
