# Chien ou chat ?

Vous trouverez ci-dessous les instructions et détails sur le programme
"Chien ou chat". Le but de se programme est de réussir à déterminer 
si une image contient un chien ou un chat ou bien de déterminer la race du chien.

Ce programme utilise le deep learning et notamment les réseaux de neurones
convolutionels (CNN), grace à la librairie PyTorch. L'entrainement des réseaux de neurones se feront sur la carte graphique de la machine pour des raisons de rapidité. 

## Installation
Ce projet utilise nécéssite un GPU (une carte graphique dédiée) pour fonctionner correctement. L'entrainement pourrait planter sur certains réseaux si vous n'avez pas assez de RAM sur la carte graphique de votre machine.

Pour installer l'application, commencez par copier le dépot suvant ([Cat_Dog_Breed sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/mustivore/Cat_Dog_Breed.git
```

Puis, accedez au dossier:

```bash
cd Cat_Dog_Breed
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer le jeu, dans l'environnement virtuel nouvellement
créé, en utilisant la commande:

```bash
poetry run python main.py
```

Vous pourrez trouver des screens de l'application dans notre rapport expliquant plus en détail le fonctionnenement de l'application.


Une fois lancé, l'application lancera une itération d'entrainement de réseau de neurone pour la classification des races de chiens. Ne vous inquiétez pas si ceci prend un certain temps. Vous pouvez aussi charger un réseau de neurone pré-entrainé pour la classification de chat/chien en cliquant sur "Select model".
Une fois le modèle selectionné vous pouvez charger une image (jpg uniquement)
en cliquant sur "Select picture(s)", puis "Predict" pour voir la prédiction 
du réseau de neurone s'afficher.


### Entrainement classification chien/chat

Pour en entrainer un nouveau réseau de neurone pour la classification chien/chat, vous pouvez utiliser le
programme `train_cat_dog.py`, utilisable comme ceci:

```bash
poetry run python train_cat_dog.py -p [path/to/file.pt] -m [model] -e [epochs]
```

l'option `-p` correspond à l'endroit où sera sauvegarder le modèle. L'extension du fichier doit être du .pt . A la fin de l'entrainement,
le modèle sera enregistré dans le dossier spécifié.

L'option `-m` correspond au type de modèle qu'on veut entrainer (alexnet, alexnet-pretrained, resnet ou resnet-pretrained). 

L'option `-e` correspond au nombre d'epoch sur lequel on veut entrainer le modèle. 

#### Avertissement

Chaque modèle enregistré dans le dossier models du projet 
sera ensuite selectionable dans l'interface de `main.py`

### Entrainement classification races de chien

Pour en entrainer un nouveau réseau de neurone pour la classification des races de chien, vous pouvez utiliser le
programme `train_dog_breed.py`, utilisable comme ceci:

```bash
poetry run python train_dog_breed.py -p [path/to/file.pt]
```

l'option `-p` correspond à l'endroit où sera sauvegarder le modèle. L'extension du fichier doit être du .pt . A la fin de l'entrainement,
le modèle sera enregistré dans le dossier spécifié.

Le programme lancera l'entrainement sur 5 epochs d'un modèle resnet pré entrainé.

[ia-gh]: https://github.com/mustivore/Cat_Dog_Breed
