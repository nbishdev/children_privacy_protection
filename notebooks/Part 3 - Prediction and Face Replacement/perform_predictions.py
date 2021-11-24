from vit_keras import vit, visualize
import tensorflow as tf
import tensorflow_addons as tfa
import os
import gdown
import shutil
import cv2
import warnings
import random
import numpy as np
from PIL import Image
from utils import *

# yoloface config
PATH_TO_YOLOFACE = './yoloface'
PATH_TO_CFG = PATH_TO_YOLOFACE + '/cfg/yolov3-face.cfg'
PATH_TO_WEIGHTS = PATH_TO_YOLOFACE + '/model-weights/yolov3-wider_16000.weights'

# application specific variables
PATH_TO_DEMO_IMGS = './demo_imgs'
PATH_TO_EMOJIS = '../dataset/emojis'
PATH_TO_MODEL = 'model.h5'


def create_directory(path):
    # create output directory if it does not already exist
    if not os.path.exists(path):
        os.makedirs(path)


def config_yoloface():
    if not os.path.isfile(PATH_TO_WEIGHTS):
        create_directory(PATH_TO_YOLOFACE + '/model-weights')
        weights_url = "https://drive.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX"
        gdown.download(weights_url, PATH_TO_YOLOFACE +
                       '/model-weights/yolov3-wider_16000.weights.zip', quiet=False)
        shutil.unpack_archive(
            PATH_TO_YOLOFACE + '/model-weights/yolov3-wider_16000.weights.zip', PATH_TO_YOLOFACE + '/model-weights/')


def get_pretrained_network():
    # create network from configuration and weights files
    net = cv2.dnn.readNetFromDarknet(PATH_TO_CFG, PATH_TO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_faces_from_image(image):
    net = get_pretrained_network()
    cap = cv2.VideoCapture(image)
    _, frame = cap.read()
    # create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    # set the network's input
    net.setInput(blob)
    # forward pass
    outs = net.forward(get_outputs_names(net))
    # remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    cap.release()
    return frame, faces


def get_cropped_face_image(frame, face):
    return frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]


def get_random_emoji():
    rdm_num = random.randint(1, 5)
    return Image.open(os.path.join(PATH_TO_EMOJIS, f'{rdm_num}.png'))


def resize_emoji(emoji, face):
    sizes = [face[2], face[3]]
    maximum_dim = max(sizes)
    return emoji.resize((maximum_dim, maximum_dim))


def classify_face(model, face):
    return model.predict(face)


def transform_image(image):
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True).standardize(image)


def center_emoji_to_face(face_loc):
    center_of_face_box = (int(
        (2 * face_loc[0] + face_loc[2]) / 2), int((2 * face_loc[1] + face_loc[3]) / 2))
    sizes = [face_loc[2], face_loc[3]]
    maximum_dim = max(sizes)
    half_dimension_of_emoji = int(maximum_dim / 2)
    return (center_of_face_box[0] - half_dimension_of_emoji, center_of_face_box[1] - half_dimension_of_emoji)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    model = tf.keras.models.load_model(PATH_TO_MODEL)
    print('[DONE] Loaded the model')
    config_yoloface()
    print('[DONE] Set yoloface configuration')
    images = [f for f in os.listdir(PATH_TO_DEMO_IMGS) if os.path.isfile(
        os.path.join(PATH_TO_DEMO_IMGS, f))]
    print(f'[DONE] Loaded {len(images)} images')
    for image in images:
        print(f'[INFO] Finding faces in image {image}')
        img_path = os.path.join(PATH_TO_DEMO_IMGS, image)
        frame, faces = get_faces_from_image(img_path)
        print(f'[INFO] Found {len(faces)} face(s) in image {image}')
        actual_img = Image.open(img_path)
        final_img = actual_img.copy()
        for face_loc in faces:
            cropped_img = get_cropped_face_image(frame, face_loc)
            resized_crop = tf.image.resize(
                tf.convert_to_tensor([cropped_img]), size=(160, 160)
            )
            prediction = classify_face(
                model, transform_image(resized_crop))[0][0]
            if prediction < 0.5:
                emoji = resize_emoji(get_random_emoji(), face_loc)
                final_img.paste(
                    emoji, center_emoji_to_face(face_loc), mask=emoji)
        # original = cv2.cvtColor(np.array(actual_img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('original image', original)
        final = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('final image', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
