# Lab — Algorithmes d'ensemble

Cohorte IA — Machine Learning Avancé.

Notebook : `lab_ensemble_cohorte.ipynb`

## Lancer

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Puis exécuter le notebook.

## Données

`make_classification` : 2500 échantillons, 20 features (12 informatives), classification binaire, split 80/20.

## Tableau de synthèse

| Modèle | Accuracy (Test) | Temps d'entraînement (s) | Observations / Surapprentissage |
|---|---:|---:|---|
| Arbre Seul | 0.816 | 0.04 | Train = 1.000. CART sans limite de profondeur → fort surapprentissage. |
| Bagging manuel | 0.898 | 1.97 | Vote majoritaire sur 50 arbres bootstrap. Variance réduite, train encore à 1.000. |
| Random Forest | 0.922 | 0.17 | Meilleur `max_features=0.3`. Limiter les variables à chaque nœud diversifie les arbres (moins corrélés qu'un bagging classique). |
| AdaBoost | 0.796 | 1.45 | Stumps, meilleur `learning_rate=1.0`. Biais encore élevé ; `lr=0.1` sous-apprend, `lr=1.5` overfitte un peu. |
| XGBoost | 0.932 | 0.32 | Log-loss test minimal à l'itération **200 / 200** : pas de surapprentissage sur 200 rounds. |
| Stacking | **0.960** | 2.23 | Méta-modèle logistique (LR + SVM + RF + XGBoost). Meilleur score test. |

Complément Partie 3 (Voting) :

| Modèle | Accuracy (Test) | Temps (s) | Observation |
|---|---:|---:|---|
| Voting (hard) | 0.924 | 0.61 | Vote sur la classe majoritaire. |
| Voting (soft) | 0.938 | 0.69 | Moyenne des probabilités, meilleur que le hard voting. |
