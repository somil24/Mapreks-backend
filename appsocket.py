from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import os
import io
import sys
import numpy as np
import cv2
import base64
from ML.object_detect.runner import runModel as obj_detect
from time import sleep
from flask_socketio import SocketIO, emit
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('connect')
def connect():
    print('\n\n\nsocket connected\n\n\n')

@socketio.on('object_frame')
def object_frame(frame):
    sbuf = StringIO()
    sbuf.write(frame)
    print(frame)

    #decode = io.BytesIO(base64.b64decode(frame))
    #img = Image.open(decode)

    #frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frame = obj_detect(frame)
    #frame = cv2.imencode('.jpg', frame)#[1]
    #frame_bytes = base64.b64encode(frame).decode('utf-8')
    #header = 'data:image/jpg;base64'
    #emit('object_frame_response', frame)

i=0
def detect_object_frames():
    # print(request.files , file=sys.stderr)
    # in deployement use request.files and then send input frame by frame
    #file = cv2.imread('./test.jpg')
    global i
    stream = './test_images/test.mp4'
    camera = cv2.VideoCapture(stream)

    while True:
        success, frame = camera.read()
        print('\n\n\nreading\n\n\n')
        if not success:
            break
        else:
            #_, buffer = cv2.imencode('.jpg', frame)
            #cv2.imwrite('output.jpg', buffer)
            buffer = obj_detect(frame)
            #buffer = cv2.imencode('.jpg', buffer)
            #sleep(3)
            print(buffer)
            #sleep(3)
            #cv2.imwrite('./output/f'+str(i)+'.jpeg', buffer)
            #i=i+1
            buffer = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')



@app.route('/detectObject', methods=['GET', 'POST'])
def detect_object():
    return Response(detect_object_frames(), mimetype='multipart/x-mixed-replace; boundry=frame')

#@app.route('/detectFace', methods=['GET'])
#def face_detect():
#    img = 


@app.route('/')
def home():
    return render_template('./index.html')


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=True)
