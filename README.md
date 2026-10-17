# film-reco

Recommandation de films content-based sur TMDB 5000, évaluée hors ligne sur les notes MovieLens.
Reprise en Python, propre et mesurée, d'un ancien projet de cours en JavaScript.

## Installation

```bash
uv venv && uv pip install -e '.[dev]'
film-reco download        # TMDB 5000 + MovieLens small dans data/ (non versionné)
```

## Utilisation

```bash
film-reco similar "the dark knight" -k 5   # recherche floue sur le titre
film-reco evaluate -k 10                   # popularité vs content-based sur MovieLens
```

`similar` affiche le score global et le détail par champ, ce qui explique chaque recommandation :

```
 tmdb_id                 title  year  score  overview  genres  keywords  cast  director
   49026 The Dark Knight Rises  2012  0.582     0.272   1.000     0.402 0.377       1.0
     272         Batman Begins  2005  0.512     0.147   0.874     0.336 0.382       1.0
```

## Fonctionnement

- **Données** (`data.py`) : TMDB 5000 (synopsis, genres, mots-clés, 5 premiers acteurs, réalisateur)
  et MovieLens `ml-latest-small`, reliés par `links.csv` (identifiant TMDB).
- **Moteur** (`content.py`) : un TF-IDF par champ, lignes normalisées. Le score entre deux films est
  la moyenne pondérée des cosinus par champ (poids dans `DEFAULT_WEIGHTS`). Un profil utilisateur est
  la moyenne des vecteurs des films qu'il a aimés.
- **Évaluation** (`evaluate.py`) : split temporel par utilisateur (20 % de notes les plus récentes en
  test), « aimé » = note ≥ 4, métriques precision@k, recall@k et couverture du catalogue, comparées à
  une référence par popularité.

## Première baseline (MovieLens small, 589 utilisateurs, 17/10/2026)

| modèle                    |  P@10  |  R@10  | couverture |
|---------------------------|--------|--------|------------|
| popularité                | 0.0591 | 0.0654 |   0.019    |
| content-based (défaut)    | 0.0290 | 0.0356 |   0.250    |
| content-based genres + kw | 0.0087 | 0.0135 |   0.269    |

La popularité gagne en précision (×2), le content-based couvre 13 fois plus du catalogue.
Le cast et le réalisateur portent l'essentiel du signal ; genres et mots-clés seuls sont faibles.
Limite connue : le split est temporel par utilisateur, mais le classement de popularité agrège les
notes de tous les utilisateurs, ce qui avantage légèrement la référence.
C'est le résultat classique : le contenu seul ne suffit pas, d'où l'intérêt d'un hybride.
Seuls 70 % des notes MovieLens portent sur un film présent dans TMDB 5000 (3 535 films).

## Feuille de route

1. Régler les poids par champ sur l'évaluation et ajouter un a priori de popularité au score.
2. Remplacer le TF-IDF du synopsis par des embeddings (sentence-transformers).
3. Ajouter un modèle collaboratif (ALS) puis un hybride, comparés avec le même protocole.

## Développement

```bash
ruff check . && ruff format --check . && pytest
```
