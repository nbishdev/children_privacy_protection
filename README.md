# Children safety and privacy protection through facial characteristics censorship in photos
## Final project in Deep Neural Networks course -- academic year 2020-2021
#### [MSc in Data Science and Information Technologies, NKUA](http://dsit.di.uoa.gr/)
### Authors:
* [Alex Zerntev](https://github.com/Alexzerntev)
* [Nikolaos Episkopos](https://github.com/nepiskopos)
* [Spyros Nikolakis](https://github.com/snikolakis)

### Description
In this project we attempt to create a facial censorship system which hides the faces of young children in images by replacing them with emojis. To this purpose we employ:
* [**YOLOFace**](https://github.com/nbishdev/yoloface) for face detection
* [**vit-keras**](https://github.com/faustomorales/vit-keras) for face classification (child / non-child)

### Dataset
The dataset we used was a custom dataset which consisted of images of people. These images were mainly harvested from the web. We also used some pictures of our families and relatives. Because of this, the dataset will NOT be shared in this repository or anywhere else.


## Project structure

### Part 1: Face detection and extraction
**Contains 1 Jupyter Notebooks**

The corresponding notebook `face_detection_extraction.ipynb` performs face extraction from the original images in our dataset.

Afterwards, the extracted faces need to be manually classified and labelled as child / non-child and afterwards split into 3 distinct datasets:
* training set
* validation set
* test set

### Part 2: Model training and evaluation
**Contains 8 Jupyter Notebooks**

The corresponding notebooks perform model training and evaluation as well as model comparison.

| Notebook name | Description |
| --- | --- |
| evaluation.ipynb | Evaluate and compare all trained models |
| resnet_fine_tune.ipynb | Fine-tune a ResNet-50 model on our data |
| resnet_train.ipynb  | Train a ResNet-50 model on our data |
| vggface_resnet_fine_tune.ipynb | Fine-tune a VGGFace (ResNet-50) model on our data |
| vggface_resnet_train.ipynb | Train a VGGFace (ResNet-50) model on our data |
| vggface_vgg_fine_tune.ipynb | Fine-tune a VGGFace (VGG16) model on our data |
| vggface_vgg_train.ipynb | Train a VGGFace (VGG16)  model on our data |
| vision_transformers.ipynb | Train a Vision Transformer model on our data |

### Part 3: Classification and face censorship
**Contains 1 Jupyter Notebook**

The corresponding notebook `prediction_censorship.ipynb` performs face detection, class prediction and children's face censorship on a subset of the original images in our dataset.


## Instructions

First of all, you need to use some Linux distribution and have `git` installed on your OS.

Set the needed parameters (paths) in every Jupyter notebook, set up your enviroment and the everything will run out of the box.

Follow the parts order when running any notebooks.


### Create the dataset

1. Create a directory named `dataset/`.
2. Under `dataset` directory create a subdirectory named `original/`.
3. Place some images containing faces inside `original`.
4. Set the path to `dataset/original/` in the `DATASET_PATH` variable both in `face_detection_extraction.ipynb` and `prediction_censorship.ipynb`.
5. You are good to go.

By default, the extracted faces will be stored under `dataset/faces_extracted/` but this can be changed by modifying the `FACES_PATH` variable in `face_detection_extraction.ipynb`.

Finally, manually choose what faces belong to infants and what does not, the first ones should be added to a directory `babies` and the second ones to a directory named `not-babies`.


### Perform predictions

In order to perform predictions and face censorship with an already trained model, you need to have a directory with some emoji icons under `dataset/` e.g. `dataset/emojis/`.

In `face_detection_extraction.ipynb` set the `DATASET_PATH`, `EMOJIS_PATH`, `MODEL_PATH` and `SAVE_PATH` to match the dataset path, the emoji directory path, the trained model path and the output directory path.

A random sample of the original images will be selected for the demo.

The results will be placed in the directory selected in `SAVE_PATH`.
