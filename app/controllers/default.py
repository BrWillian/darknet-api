from app import app
from flask import request
from numpy import fromstring, uint8, argmax
from base64 import b64decode, b64encode
from cv2 import imdecode, IMREAD_COLOR, imencode
from cv2.dnn import readNetFromDarknet, blobFromImage


@app.route('/api', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        try:
            imgb64 = request.data['image']
            img = fromstring(b64decode(imgb64), uint8)
            img = imdecode(img, IMREAD_COLOR)
            (H, W) = img.shape[:2]

            net = readNetFromDarknet('app/yolo_cfg/yolov3-custom.cfg', 'app/yolo_cfg/yolov3.weights')
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            blob = blobFromImage(img, 1/255.5, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = argmax(scores)
                    confidences = scores[classID]

            return {'image': 'ok'}

        except NameError as erro:
            return {'image': erro}

    return {'message': 'error!'}
