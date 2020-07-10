from app import app
from flask import request
from numpy import fromstring, uint8, argmax, array
from base64 import b64decode, b64encode
from cv2 import imdecode, IMREAD_COLOR, imencode, rectangle, putText, FONT_HERSHEY_SIMPLEX
from cv2.dnn import readNetFromDarknet, blobFromImage, NMSBoxes


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
                    confidence = scores[classID]

                    if confidence > 0.5:
                        box = detection[0:4] * array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = NMSBoxes(boxes, confidences, 0.5, 0.4)
            labels = open('app/yolo_cfg/display.names').read().strip().split('\n')
            colors = [[0, 255, 0], [0, 0, 255]]

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in colors[classIDs[i]]]
                    rectangle(img, (x-2, y-2), (x+w+2, y+h+2), color, 1)
                    text = '{}'.format(labels[classIDs[i]])
                    acc = '{:.2f}%'.format(confidences[i]*100)
                    rectangle(img, (x-2, y+62), (x+47, y+47), color, -1)
                    putText(img, text, (x-2, y+60), FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    putText(img, acc, (x-2, y+77), FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            _, pred = imencode('.jpg', img)
            pred = pred.tobytes()
            pred = b64encode(pred)

            try:
                boolean = "false" if classIDs[0] else "true"
            except:
                boolean = "null"
            return {
                'nomeArquivo': request.data['nomeArquivo'],
                'image': pred.decode('utf-8'),
                'pred': boolean
            }

        except:
            return {'image': "Imagem invalida!"}

    return {'message': 'Darknet Api Vizentec'}


@app.route('/api-boolean', methods=['POST', 'GET'])
def bolean():
    if request.method == "POST":
        try:
            imgb64 = request.data['image']
            img = fromstring(b64decode(imgb64), uint8)
            img = imdecode(img, IMREAD_COLOR)
            net = readNetFromDarknet('app/yolo_cfg/yolov3-custom.cfg', 'app/yolo_cfg/yolov3.weights')
            blob = blobFromImage(img, 1 / 255.5, (416, 416), swapRB=True, crop=False)
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        classIDs.append(classID)
            try:
                boolean = "false" if classIDs[0] else "true"
            except:
                boolean = "null"
            return {
                'nomeArquivo': request.data['nomeArquivo'],
                'pred': boolean
            }

        except:
            return {'image': "Imagem invalida!"}

    return {'message': 'Darknet Api Vizentec'}