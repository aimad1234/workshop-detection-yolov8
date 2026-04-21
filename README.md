# workshop-detection-yolov8
# 🍊 Détection de Fruits avec YOLOv8 — Séance 2

> **Notebook pédagogique** destiné à des participants débutants en Machine Learning, appliqué à la détection automatique de fruits sur arbres fruitiers à partir de photos de terrain.

---

## 🎯 Contexte du projet

Estimer le rendement d'une parcelle fruitière est une tâche longue et coûteuse lorsqu'elle est réalisée manuellement. Ce projet propose d'automatiser ce processus grâce à la **détection d'objets** par deep learning : à partir d'une photo d'arbre, un modèle YOLOv8 localise et compte automatiquement les fruits visibles, en distinguant ceux encore sur l'arbre de ceux tombés au sol.

Le dataset utilisé est le [CitDet (Citrus Detection)](https://www.kaggle.com/datasets/andresmgs/citdet-yolo), disponible sur Kaggle, qui contient des images annotées d'agrumes avec deux classes :
- 🌳 `fruit_on_tree` — fruit sur l'arbre
- 🍊 `fruit_on_ground` — fruit tombé au sol

---

## 🔄 Lien avec la Séance 1

Ce notebook fait suite à la **Séance 1** (classification des maladies des plantes avec SVM et Random Forest). Il introduit un niveau de complexité supplémentaire :

| | Séance 1 (ML) | Séance 2 (DL) |
|---|---|---|
| **Tâche** | Classification | Détection d'objets |
| **Algorithmes** | SVM, Random Forest | YOLOv8 (deep learning) |
| **Sortie** | Une classe par image | Boîtes englobantes + classes |
| **Features** | Extraites manuellement | Apprises automatiquement |
| **Question** | "Quelle maladie ?" | "Où sont les fruits ? Combien ?" |

---

## 🎓 Objectifs de la formation

À l'issue de ce notebook, les participants sont capables de :

1. **Distinguer** classification et détection d'objets
2. **Comprendre** l'architecture YOLO et le principe du transfer learning
3. **Configurer** un dataset au format YOLO (fichier `.yaml`, annotations `.txt`)
4. **Entraîner** un modèle YOLOv8 sur un dataset personnalisé
5. **Interpréter** les métriques de détection : Précision, Rappel, mAP@0.5, F1
6. **Visualiser** les prédictions avec des boîtes englobantes
7. **Appliquer** le modèle à une estimation concrète de rendement agricole

---

## 📋 Contenu du notebook

Le notebook est structuré en 12 étapes progressives :

| Étape | Contenu | Concepts abordés |
|-------|---------|-----------------|
| 1 | Installation des dépendances | `ultralytics`, `kagglehub`, `opencv` |
| 2 | Imports | Présentation de l'écosystème deep learning |
| 3 | Téléchargement du dataset | Format YOLO, structure des annotations |
| 4 | Exploration des données | Structure train/val/test, comptage |
| 5 | Fichier de configuration YAML | Description du dataset pour YOLOv8 |
| 6 | Paramètres d'entraînement | Epochs, batch size, taille du modèle |
| 7 | Chargement du modèle | Transfer learning, poids pré-entraînés COCO |
| 8 | Entraînement | Loss, rétropropagation, GPU vs CPU |
| 9 | Évaluation | mAP, IoU, Précision, Rappel, F1 |
| 10 | Prédiction & visualisation | Bounding boxes, seuil de confiance |
| 11 | Test sur images aléatoires | Inférence sur le jeu de test |
| 12 | Bilan & pistes d'amélioration | Synthèse, déploiement, ressources |

---

## 🔧 Prérequis

### Connaissances

- Bases de Python (variables, fonctions, boucles)
- Avoir suivi la Séance 1 (ou notions équivalentes en ML)
- Aucune connaissance préalable en deep learning n'est requise

### Environnement recommandé

Ce notebook a été conçu pour s'exécuter sur **[Kaggle](https://www.kaggle.com/)** avec le GPU activé.  
Il peut également tourner sur Google Colab (GPU T4 gratuit) ou en local avec une carte graphique NVIDIA.

> ⚠️ Un GPU est fortement recommandé. Sans GPU, l'entraînement sur 50 epochs peut prendre plusieurs heures.

### Bibliothèques requises

```
ultralytics>=8.0
torch>=2.0
opencv-python>=4.8
matplotlib>=3.6
kagglehub
PyYAML
```

---

## 🚀 Instructions pour exécuter le notebook

### Option 1 — Kaggle (recommandé)

1. Créez un compte gratuit sur [kaggle.com](https://www.kaggle.com/)
2. Créez un nouveau notebook et importez le fichier `.ipynb`
3. Activez le GPU : **Settings → Accelerator → GPU T4 x2**
4. Ajoutez le dataset : **Add Data → andresmgs/citdet-yolo**
5. Exécutez toutes les cellules dans l'ordre (**Run All**)

### Option 2 — Google Colab

1. Ouvrez [colab.research.google.com](https://colab.research.google.com/)
2. Importez le fichier `.ipynb` via **Fichier → Importer un notebook**
3. Activez le GPU : **Exécution → Modifier le type d'exécution → GPU**
4. Installez les dépendances (première cellule) puis adaptez `DATA_PATH`

### Option 3 — Local (avec GPU NVIDIA)

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/<votre-utilisateur>/crop-disease-detection-ml.git
   cd crop-disease-detection-ml/seance2
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez le dataset depuis Kaggle et placez-le dans `./data/CidDet-YOLO/`

4. Modifiez le champ `path` dans `citdet_data.yaml` :
   ```yaml
   path: ./data/CidDet-YOLO
   ```

5. Lancez Jupyter :
   ```bash
   jupyter notebook citdet_yolov8_pedagogique.ipynb
   ```

---

## 📁 Structure du dépôt

```
seance2/
├── citdet_yolov8_pedagogique.ipynb   # Notebook principal (version pédagogique)
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
└── citdet_data.yaml                   # Fichier de configuration du dataset (généré par le notebook)
```

---

## 📦 requirements.txt

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
matplotlib>=3.6.0
PyYAML>=6.0
kagglehub>=0.1.0
```

---

## 📊 Résultats obtenus

Avec YOLOv8n entraîné sur 50 epochs (GPU Kaggle) :

| Métrique | Valeur typique |
|----------|---------------|
| mAP@0.5 | ~0.75 – 0.85 |
| Précision | ~0.80 – 0.90 |
| Rappel | ~0.70 – 0.85 |
| F1-Score | ~0.75 – 0.87 |

> Les résultats varient selon le nombre d'epochs, la taille du modèle et la résolution des images.

---

## 🔬 Dataset

- **Source** : [Kaggle — CitDet YOLO (andresmgs)](https://www.kaggle.com/datasets/andresmgs/citdet-yolo)
- **Contenu** : photos d'arbres fruitiers (agrumes) annotées au format YOLO
- **Classes** : `fruit_on_tree` (0), `fruit_on_ground` (1)
- **Format des annotations** : fichiers `.txt` avec `classe cx cy largeur hauteur` (valeurs normalisées)

---

## 💡 Pistes d'amélioration

- **Plus d'epochs** (100–200) pour un meilleur apprentissage
- **Modèle plus grand** (`yolov8s`, `yolov8m`) si GPU disponible
- **Réduction du seuil de confiance** pour détecter plus de fruits (au prix de plus de faux positifs)
- **Export du modèle** en ONNX ou TensorRT pour le déploiement en production
- **Application mobile** : intégrer le modèle dans une app de terrain via TFLite ou CoreML

---

## 📚 Ressources

- [Ultralytics YOLOv8 — Documentation officielle](https://docs.ultralytics.com/)
- [Dataset CitDet sur Kaggle](https://www.kaggle.com/datasets/andresmgs/citdet-yolo)
- [Comprendre mAP — Article Medium](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Tutoriel YOLOv8 sur dataset custom — Roboflow](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [Introduction au Transfer Learning](https://cs231n.github.io/transfer-learning/)

---

## 📝 Licence

Ce projet est partagé à des fins pédagogiques.  
Le dataset CitDet est soumis aux conditions d'utilisation de Kaggle.

---

*Formation Machine Learning appliqué à l'Agriculture — Séance 2 — 2026*
