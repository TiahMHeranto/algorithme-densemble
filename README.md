# Algorithmes d'ensemble

*Weak learners* agrégés pour réduire la **variance** (Bagging / Random Forest) ou le **biais** (AdaBoost / XGBoost), puis combinés par **vote** et **stacking**.

## Jeu de données

Classification binaire synthétique (`sklearn.datasets.make_classification`) :

- 2500 échantillons, split 80 / 20 stratifié
- 20 features dont 12 informatives (4 redondantes)
- `random_state=42` pour la reproductibilité

## Lancer le lab

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Le notebook `lab_ensemble_cohorte.ipynb` reprend la même démarche, partie par partie.

## Méthodes

| Partie | Méthode | Ce qui a été fait |
|---|---|---|
| 1.1 | CART sans régularisation → surapprentissage | `DecisionTreeClassifier` profondeur illimitée |
| 1.2 | Bootstrap + Aggregation, vote majoritaire, OOB | Bagging manuel, **B = 50**, erreur OOB |
| 1.3 | Random Forest = bagging + `m` features tirées à chaque nœud | Variation de `max_features`, importances MDI |
| 2.1 | AdaBoost : stumps, repondération, `learning_rate` | 200 stumps, grille `lr ∈ {0.1, 0.5, 1.0, 1.5}` |
| 2.2 | Gradient boosting / XGBoost, shrinkage, log-loss | `eval_set`, courbe Log-Loss, détection du surapprentissage |
| 3 | Hard voting, soft voting (proba des feuilles), métamodèle | LR + SVM calibré + RF + XGBoost, stacking logistique |

**Pourquoi `max_features` crée de la diversité.**  
Si toutes les features sont candidates à chaque split (Bagging classique), les arbres choisissent souvent la même variable dominante et restent **corrélés**. Restreindre le split à un sous-ensemble aléatoire (`sqrt`, `log2`, …) force des coupes différentes : la corrélation $\rho$ baisse, donc la variance du vote $\mathrm{Var}(\bar h)=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$ aussi.

## Tableau de synthèse

Métriques mesurées sur le jeu de test (500 observations). Les temps sont ceux du `fit` uniquement.

| Modèle | Accuracy (Test) | Temps d'entraînement (s) | Observations / Surapprentissage |
|---|---:|---:|---|
| Arbre Seul | 0.794 | 0.04 | Train = **1.000**. CART non borné isole le train ; écart train–test de 0.206. |
| Bagging manuel | 0.878 | 1.69 | Vote majoritaire sur 50 arbres bootstrap. OOB acc. = 0.893. Variance réduite, écart encore de 0.122. |
| Random Forest | 0.894 | 0.21 | Meilleur `max_features=sqrt` (équivalent `log2` ici). Arbres moins corrélés que le bagging (`max_features=None` → 0.880). Extra Trees : 0.898. |
| AdaBoost | 0.770 | 1.45 | Stumps (profondeur 1), `lr=1.5`. Le biais des stumps reste élevé sur 12 features interactives : moins bon que l'arbre profond. `lr=0.5` est plus stable (test 0.766, moins d'écart). |
| XGBoost | 0.900 | 0.33 | Arbres de profondeur 3 + shrinkage 0.1. Log-loss test minimal à l'**itération 195 / 200**, puis légère remontée (début de surapprentissage boosting). |
| Stacking | **0.928** | 2.19 | Méta-modèle logistique sur les prédictions CV. Meilleur test, plus petit écart train–test (0.050). |

Complément Voting :

| Modèle | Accuracy (Test) | Temps (s) | Observation |
|---|---:|---:|---|
| Voting (hard) | 0.896 | 4.09 | Classe majoritaire. |
| Voting (soft) | 0.910 | 0.69 | Moyenne des probabilités, plus stable que le hard voting. |


## Choix de modèle

1. **Baseline bagging / forêt** en premier : peu de surapprentissage quand on ajoute des arbres, bon plancher.
2. **XGBoost** ensuite si le biais reste trop haut (ici : AdaBoost-stumps sous-apprend, XGBoost passe à 0.900).
3. Ajouter des arbres en RF **réduit** la variance ; en boosting, trop d'arbres **sans** baisser le `learning_rate` fait overfitter.
4. **Stacking** quand on dispose déjà de modèles hétérogènes et que le coût CV est acceptable — ici le meilleur score test.

Détail numérique : `results/benchmark.csv`.
