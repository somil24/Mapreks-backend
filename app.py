from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
from PIL import Image
import os
import collections
import tensorflow as tf

import io
import sys
import numpy as np
import base64
import cv2
from generators.opt_face_detection_generator import detect_face_frames
from time import sleep
from generators.utils import (
    load_model,
    get_faces_live,
    forward_pass,
    load_embeddings,
    save_image
)
import pickle

from generators.lib.mtcnn import detect_face
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

pnet = None
onet = None
rnet = None

Sess_dict = {'sess': None, 'img_placehoder': None, 'emb': None, 'phasetrain_placeholder': None}

i=0

@socketio.on('connect')
def init():
    global i
    i=0
    print('[info] new socket connected')

@socketio.on('detectFace')
def handle_message(data):
    global pnet, onet, rnet, Sess_dict, i
        
    print("recieved frame: ", data['frame'], "\n and processing frame ", i)
    i=i+1
    # try:
    img_data = np.fromstring(data['data'], dtype=np.uint8)
    classifier_path = '../public/60b8f9032a0809472eccffe5/classify_model.pkl'
    img = cv2.imdecode(img_data, 1)
    ret_boxes, frame = detect_face_frames(img, Sess_dict['sess'], Sess_dict['img_placehoder'],
        Sess_dict['emb'],Sess_dict['phasetrain_placeholder'], pnet, rnet, onet, classifier_path)

    print('\n\n==>', ret_boxes)
    # if idxs>0:        
    socketio.emit('detectedFace', {'faces': ret_boxes, 'frame': i, 'image': frame})


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    # response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


def boot_face_detector():
    global pnet, onet, rnet, Sess_dict

    PATH='./generators'

    print('Loading feature extraction model')
    FACENET_MODEL_PATH = PATH+'/model/20170512-110547/20170512-110547/20170512-110547.pb'
    facenet_model=load_model(FACENET_MODEL_PATH)

    tf.compat.v1.disable_eager_execution()
    config = tf.compat.v1.ConfigProto(device_count={'GPU':0, 'CPU':2})
    print('\n\n\ncreating sessions\n\n\n')
    # Get input and output tensors
    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    # Initiate persistent FaceNet model in memory
    facenet_persistent_session = tf.compat.v1.Session(graph=facenet_model, config=config)
    # facenet_persistent_session = tf.compat.v1.Session(graph=facenet_model, config=config, log_device_placement=True)
    Sess_dict = {'sess': facenet_persistent_session, 'img_placehoder': images_placeholder, 'emb': embeddings, 'phasetrain_placeholder': phase_train_placeholder}
    # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

if __name__ == '__main__':
    print('[INFO] booting face detector')
    boot_face_detector()
    print('[INFO] complete booting face detector')
    socketio.run(app, host='0.0.0.0')


# [
#     {x, y. w. h}
#     {x, y. w. h}
#     {x, y. w. h}
#     {x, y. w. h}
# ]