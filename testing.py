from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import ML.facenet.src.facenet as facenet
import os
import sys
import math
import pickle
import ML.facenet.src.align.detect_face as detect_face
import numpy as np
import cv2
import collections
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import scipy
import imageio
import time
from time import sleep

PATH='./ML/facenet/src'

def run_facenet_model(frame):
    i=0 
    j=0
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    FACENET_MODEL_PATH = PATH+'/model/20170512-110547/20170512-110547.pb'
    CLASSIFIER_PATH = PATH+'/model/best_model2.pkl'
    OUT_ENCODER_PATH = PATH+'/model/out_encoder.pkl'
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    # Load the out encoder 
    with open(OUT_ENCODER_PATH, 'rb') as file:
        out_encoder = pickle.load(file)
    print("Out Encoder, Successfully loaded")


    tf.compat.v1.disable_eager_execution()

    with tf.Graph().as_default():
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU':0}, log_device_placement=True))
    	
        with sess.as_default():
    	    # Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = detect_face.create_mtcnn(sess, PATH+"/align")

            people_detected = set()
            person_detected = collections.Counter()
            
    
                #if frame.ndim == 2:
                #    frame = facenet.to_rgb(frame)
                #sleep(8)
                #print("\t\t\t\t\t")
                #print("")
                #print(frame)
                #print("\n\n\n\n")
                #sleep(2)
            bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            
            faces_found = bounding_boxes.shape[0]
            #sleep(8)
            print("\t\t\t\t\t\t\t\n\n\n\n\n\n")
            print("faces found: ", faces_found)
            print("\n\n\n\\n\n\n")
            #print(frame)
            #cv2.imwrite('./input/f'+ str(i) +'.jpeg', frame)
            #i=i+1
            try:
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                    
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        #print("\t\t\t\t\t\t\t\n\n\n\n\n\n")
                        #print("FRAME SHAPE: ", scaled_reshape.shape)
                        #print("\n\n\n\\n\n\n")
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        #print("\t\t\t\t\t\t\t\n\n\n\n\n\n")
                        #print("EMBEDDINGS SHAPE: ", emb_array.shape)
                        #print("\n\n\n\\n\n\n")
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = out_encoder.inverse_transform(best_class_indices)
                        best_name = best_name[0]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                    
                        
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20  
                        
                        if best_class_probabilities > 0.25:
                            name = best_name
                        else:
                            name = "Unknown"
                        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                        person_detected[best_name] += 1
                    #writeTo = './frames/f'+i+'.jpeg'
                        #x = cv2.imwrite('frames.jpeg', frame)
                        #print('x=', x)
                        #i=i+1
                    #writeTo = 
                    #cv2.imwrite('./output/f'+str(j)+'.jpeg', frame)
                    #j=j+1

            except Exception as e:
                print('in except: {}'.format(e))
                sleep(8)
                pass
                    
                #cv2.imshow('Face Recognition',frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                #_, buffer = cv2.imencode('.JPEG', frame)
                #buffer=np.squeeze(buffer)
                #imageio.imwrite(uri=PATH+'/output/test.jpeg', im=buffer)
            cv2.imwrite('./test/t'+str(i)+'.jpg', frame)
            i=i+1
            return frame
            # yield(b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #cv2.imwrite('output.jpeg', frame)
            
            #cap.release()
            #cv2.destroyAllWindows()

frame = cv2.imread('./test_images/astest.jpg')
run_facenet_model(frame)
