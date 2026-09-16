# Creole_Approche_hybride

## Système de traduction automatique pour le créole haïtien

Ce dépôt contient les scripts, les configurations et les résultats expérimentaux réalisés dans le cadre de mon mémoire de maîtrise en informatique, avec une spécialisation en intelligence artificielle, à l’Université du Québec à Montréal (UQAM).

L’objectif de ce projet est de développer et d’évaluer des systèmes de traduction automatique pour le créole haïtien, principalement vers l’anglais et le français. Le projet explore différentes stratégies d’amélioration des performances, notamment l’intégration de données culturelles et l’utilisation de données synthétiques avec alternance codique.

## Structure du dépôt

```text
Creole_Approche_hybride/
│
├── datasets/              # Données utilisées pour les expérimentations
├── fairseq/               # Scripts liés aux modèles Transformer avec Fairseq
├── nllb200/               # Scripts liés au fine-tuning du modèle NLLB-200
├── cultural_nllb/         # Expériences avec les données culturelles
├── cultural_oral/         # Expériences avec le corpus de l’Atlas Linguistique d’Haïti(ALH)
├── code_switching/        # Données et scripts liés à l’alternance codique
├── mt5/                   # Expériences liées au modèle mT5
├── LLM/                   # Scripts liés à l’utilisation de modèles de langue
├── result/                # Résultats générés
├── results/               # Résultats d’évaluation
├── stats/                 # Statistiques sur les données et les expériences
├── uqam_eval_srcs/        # Scripts d’évaluation utilisés dans le cadre du Shared Task
├── config.py              # Fichier de configuration
├── command.txt            # Commandes utilisées pour les expériences
└── README.md              # Présentation du projet
```

## Données utilisées

Les expérimentations reposent sur plusieurs types de données :

* des données générales issues de corpus de traduction automatique; 
* des données culturelles liées à la langue et à la culture haïtiennes ;
* des données synthétiques d’alternance codique ;
* des données utilisées dans le cadre du WMT 2026 Creole Machine Translation Shared Task.

Certaines données ne sont pas incluses directement dans ce dépôt pour des raisons de taille.

## Modèles utilisés

Deux principales approches sont comparées :

1. **Transformer entraîné à partir de zéro avec Fairseq**
   
   Ce modèle est entraîné directement sur les corpus préparés pour les différentes expérimentations.

2. **NLLB-200**
   
   Le modèle multilingue préentraîné NLLB-200 est adapté aux données du projet par fine-tuning.

## Métriques d’évaluation

Les performances des systèmes sont évaluées à l’aide de plusieurs métriques, notamment :

* BLEU ;
* chrF++ ;
* TER ;
* BLEURT ;
* CSI-Match ;
* Evaluation humaine ;

## Prétraitement des données

Les données sont préparées sous deux formats principaux :

* fichiers `.jsonl` pour les expériences avec NLLB-200 ;
* fichiers `.ht`, `.en`, `.fr` et `.spm` pour les expériences avec Fairseq.

Les scripts de prétraitement permettent de nettoyer, segmenter, combiner et convertir les corpus selon le format requis par chaque modèle.

## Entraînement et évaluation

Les expériences sont organisées selon plusieurs configurations :

* modèle de référence entraîné sur les données générales de Kreyòl-MT ;
* modèle enrichi avec des données culturelles : Kreyòl-MT + données culturelles ;
* modèle enrichi avec des données culturelles et des données d’alternance codique : Kreyòl-MT + données culturelles + données d’alternance codique.

Les résultats sont ensuite comparés afin d’évaluer l’effet de chaque stratégie d’enrichissement des données.


## Installation

1. Cloner le dépôt ou télécharger le fichier ZIP :

```bash
git clone https://github.com/<votre-utilisateur>/Creole_Approche_hybride.git
```

2. Accéder au dossier du projet :

```bash
cd Creole_Approche_hybride
```

3. Créer un environnement virtuel :

```bash
python -m venv env
```

4. Activer l’environnement virtuel.

Sous Windows :

```bash
env\Scripts\activate
```

Sous Linux/macOS :

```bash
source env/bin/activate
```

5. Installer les dépendances nécessaires :

## Pour les expériences avec Fairseq :

```bash
pip install -r fairseq_requirement.txt
```

## Pour les expériences avec NLLB-200 :

```bash
pip install -r nllb_requirement.txt
```

## Utilisation

Les scripts du projet permettent de préparer les données, d’entraîner les modèles et d’évaluer les résultats obtenus.

Avant d’exécuter les scripts, il faut s’assurer que :

* l’environnement virtuel est activé ;
* les données nécessaires sont placées dans le dossier `datasets/` ;
* les chemins définis dans les scripts ou dans le fichier de configuration sont correctement adaptés à l’environnement local.

Les principales étapes d’utilisation sont les suivantes :

1. préparer les données au format requis ;
2. lancer l’entraînement du modèle choisi ;
3. générer les prédictions ;
4. évaluer les traductions à l’aide des métriques automatiques.

Les commandes exactes utilisées pour les expériences sont documentées dans le fichier `command.txt`.

## Contribution

Les contributions sont les bienvenues. Les utilisateurs peuvent proposer des améliorations, signaler des problèmes ou soumettre des corrections en ouvrant une issue ou une pull request sur le dépôt GitHub.

## Licence

Ce projet est fourni à des fins de recherche et de documentation scientifique.

## Remerciements

Ce projet a été réalisé dans le cadre d’un mémoire de maîtrise portant sur la traduction automatique du créole haïtien.

Nous remercions les personnes et organismes ayant contribué à la collecte, à l’accès ou à la valorisation des ressources linguistiques et culturelles utilisées dans ce travail.



## Citation

Si vous utilisez les ressources dans vos travaux de recherche, veuillez citer cet article :

```bibtex
@inproceedings{bellune2026culturalawareness,
  author    = {Bellune, Tabitha Megane and Le, Ngoc Tan and Sadat, Fatiha},
  title     = {{Evaluating and Improving Cultural Awareness in Haitian Creole Machine Translation}},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation (WMT 2026)},
  year      = {2026},
  address   = {Budapest, Hungary},
  month     = oct,
  note      = {Accepted}
}
```
