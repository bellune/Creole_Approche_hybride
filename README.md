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
