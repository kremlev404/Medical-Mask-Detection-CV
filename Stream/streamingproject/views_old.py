from django.shortcuts import render
from django.http import  StreamingHttpResponse, HttpResponseServerError, HttpResponseRedirect
from django.views.decorators import gzip
from django.shortcuts import redirect
import cv2
from time import time
from math import sqrt, floor
import numpy as np

myauth = False
classesFile = "classes.names"
ID = 1

# Считываем названия классов
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Считываются все данные для yolo
yolo_net = cv2.dnn.readNetFromDarknet("yolov3.cfg",
                                      'yolo-obj_final.weights')
renet = cv2.dnn.readNet('person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml',
                        'person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.bin')
net = cv2.dnn.readNet('person-detection-retail-0013/FP32/person-detection-retail-0013.xml',
                      'person-detection-retail-0013/FP32/person-detection-retail-0013.bin')
vino_face_net = cv2.dnn.readNet("face-detection-retail-0005/FP32/face-detection-retail-0005.xml",
                                "face-detection-retail-0005/FP32/face-detection-retail-0005.bin")

vino_face_net_size = (300, 300)
netsize = (544, 320)
renetsize = (64, 160)

# Минимальная вероятность для лица - 90 процентов
face_threshold = 0.9

inpWidth = 608
inpHeight = 608
step = 1

person_threshold = 0.9
distance_threshold = 0.4
mask_threshold = 0.5
nms_threshold = 0.1

yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
renet.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
renet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

video = cv2.VideoCapture(0)

# Массив с координатами лиц на видео
cropped_faces = []
distance = 0
chips = []
times = {}
data = {}
fps = 1
trigger = 0.8
dist_trigger = 0.3
f_trigger = 0.5


def compare(data, chip, save=0):
    reblob = cv2.dnn.blobFromImage(chip, size=renetsize, ddepth=cv2.CV_8U)
    renet.setInput(reblob)
    reout = renet.forward()
    reout = reout.reshape(256)
    reout /= sqrt(np.dot(reout, reout))
    ide = 1
    distance = -1

    if len(data) != 0:
        for x in data:
            distance = np.dot(reout, data[x])
            ide += 1
            if distance > dist_trigger:
                ide = x
                break

    if distance < dist_trigger:
        data['id{}'.format(ide)] = reout
        if save:
            cv2.imwrite('photos/id{}.jpg'.format(ide), chip)

    return distance, ide


# Функция рисует боксы для масок на кадре
def yolo_postprocess(frame, outs):
    # Размеры кадра
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []

    confidences = []

    boxes = []

    # out - массив выходных данных из одного слоя ОДНОГО кадра(всего слоев несколько)
    for out in outs:
        # detection - это один распознанный на этом слое объект
        for detection in out:

            # извлекаем ID класса и вероятность
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Если "уверенность" больше минимального значения, то находим координаты боксы
            if confidence > mask_threshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                # Обновим все три ранее созданных массива
                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # сейчас имеем заполненные массивы боксов и ID для одного кадра
    # применим non-maxima suppression чтобы отфильтровать накладывающиеся ненужные боксы
    # для КАЖДОГО кадра indices обновляются
    indices = cv2.dnn.NMSBoxes(boxes, confidences, mask_threshold, nms_threshold)

    for i in indices:
        # То есть мы "отфильтровали" накладывающиеся боксы и сейчас СНОВА получаем координаты уже
        # отфильтрованных боксов
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        width = box[2]  # Это именно ШИРИНА,а не координата x правого нижнего угла
        height = box[3]
        mask_box_coords = [x, y, width, height]

        # Название класса
        label = '%.2f' % confidences[i]
        # Получаем название класса и "уверенность"
        if classes:
            assert (classIDs[i] < len(classes))
            label = '%s:%s' % (classes[classIDs[i]], label)

        # Рисуем бокс и название класса
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, labelSize[1])
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return mask_box_coords


# Функция выводит название только необходимы нам слоев
def getOutputsNames(net):
    # Выводим названия всех слоёв в сетке
    layersNames = net.getLayerNames()

    # Выводим названия только слоев с несоединенными выводами (?)
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# ------------------------------------------------------------------------------------------------------------

# Функция рисует боксы для лиц на кадре и заполняет внешний массив координат лиц
# Также проверяется, надета ли маска
def vino_face_postprocess(frame, outs, mask_box_coords):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    for detection in outs.reshape(-1, 7):
        confidence = float(detection[2])
        x = int(detection[3] * frame.shape[1])
        y = int(detection[4] * frame.shape[0])
        width = int(detection[5] * frame.shape[1]) - x  # Это именно ширина, а не координтата нижнего правого угла
        height = int(detection[6] * frame.shape[0]) - y

        # Получаем координаты лица
        face_box_coords = [x, y, x + width, y + height]

        if confidence > face_threshold:
            cropped_face = frame[y:height, x:width]
            # Координаты лица добавляются в массив
            cropped_faces.append(cropped_face)
            # Название класса
            label = '%.2f' % confidence
            label = '%s:%s' % ("face", label)

            # Изначально будем считать, что маска не надета
            mask_inside = False
            status = "No mask"
            status_color = (0, 0, 255)

            # Проверяем, находится ли центр маски внутри бокса лица и меняем цвет бокса маски
            if (face_box_coords != []) and (mask_box_coords != []) and (face_box_coords is not None) and (
                    mask_box_coords is not None):
                mask_inside = check_if_mask_inside_face(face_box_coords, mask_box_coords)

            # Если маска надета, то лицо рисуется зеленым, иначе - красным
            if mask_inside:
                color = (0, 255, 0)
                status = "Mask is on"
                status_color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Статус маски пишется в правом нижнем углу
            cv2.putText(frame, status, (int(frame_width / 2) - 40, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        status_color, 2)

            # Рисуем бокс лица и название
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(y, labelSize[1])
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return face_box_coords


# Функция сравнивает два найденных бокса человека
def vino_person_compare(data, box):
    vino_person_re_blob = cv2.dnn.blobFromImage(box, size=vino_person_re_net_size, ddepth=cv.CV_8U)
    vino_person_re_net.setInput(vino_person_re_blob)
    vino_person_re_outs = vino_person_re_net.forward()
    vino_person_re_outs = vino_person_re_outs.reshape(256)
    vino_person_re_outs /= sqrt(np.dot(vino_person_re_outs, vino_person_re_outs))
    ide = 1
    distance = -1

    if len(data) != 0:
        for x in data:
            distance = np.dot(vino_person_re_outs, data[x])
            ide += 1
            if distance > distance_threshold:
                ide = x
                break

    if distance < distance_threshold:
        data['id{}'.format(ide)] = vino_person_re_outs

    return distance, ide


# Функция рисует боксы для человека, а также отслеживает человека
def vino_person_postprocess(frame, outs):
    data = {}
    objects = 0
    for detection in outs.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        # Получаем ID человека

        if confidence > person_threshold:
            objects += 1
            box = frame[ymin:ymax, xmin:xmax]

            # Рисуется первый бокс человека
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 215, 0), 2)

            # Пробуем сравнить два бокса человека, чтобы отследить его передвижение
            try:
                distance, ID = vino_person_compare(data, box)
            except:
                continue

            # На кадре рисуется ID человека
            cv2.putText(frame, 'person {}'.format(ID), (xmin, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                        1)

    # Возвращаем координаты бокса человека
    person_box_coords = [xmin, ymin, xmax, ymax]
    return person_box_coords


# ------------------------------------------------------------------------------------------------------------
# Функция проверяет, находится ли центр бокса маски в пределах бокса лица
def check_if_mask_inside_face(face_box_coords, mask_box_coords):
    face_x = face_box_coords[0]
    face_y = face_box_coords[1]
    face_width = face_box_coords[2]
    face_height = face_box_coords[3]

    mask_x = mask_box_coords[0]
    mask_y = mask_box_coords[1]
    mask_width = mask_box_coords[2]
    mask_height = mask_box_coords[3]

    # Получаем координаты середины бокса маски
    mask_center_x = int(floor(mask_x + (mask_x + mask_width)) / 2)
    mask_center_y = int(floor(mask_y + (mask_y + mask_height)) / 2)

    # Это массивы координат для проверки
    face_hor = range(face_x, face_x + face_width)
    face_vert = range(face_y, face_y + face_height)

    # Если координты центра маски есть в обоих массивах, то маска надета
    if (mask_center_x in face_hor) and (mask_center_y in face_vert):
        return True

    return False


# ------------------------------------------------------------------------------------------------------------
class VideoCamera():
    step = 2

    def get_frame(self, count):
        start = time()
        grab, frame = video.read()
        if not grab:
            raise Exception('Image not found!')

        # frame = cv2.resize(frame1, (608,608), interpolation = cv2.INTER_AREA)
        if count % self.step == 0:
            vino_face_blob = cv2.dnn.blobFromImage(frame, size=vino_face_net_size, ddepth=cv2.CV_8U)
            vino_face_net.setInput(vino_face_blob)

            blob = cv2.dnn.blobFromImage(frame, size=netsize, ddepth=cv2.CV_8U)
            net.setInput(blob)
            out = net.forward()

            yolo_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            yolo_net.setInput(yolo_blob)
            yolo_outs = yolo_net.forward(getOutputsNames(yolo_net))

            # Получаем выходные данные c vino_face
            vino_face_outs = vino_face_net.forward()

            # Получаем выходные данные c vino_person
            vino_person_outs = net.forward()

            # Данные с yolo обрабатываются в функции
            mask_box_coords = yolo_postprocess(frame, yolo_outs)

            # Данные с vino_face обрабатываются в функции
            face_box_coords = vino_face_postprocess(frame, vino_face_outs, mask_box_coords)

            # Данные с vino_person обрабатываются в функции
            person_box_coords = vino_person_postprocess(frame, vino_person_outs)

        # Завершаем отсчёт времени работы для вычисления FPS
        end = time()
        fps = 1 / (end - start)

        cv2.putText(frame, 'fps:{:.2f}'.format(fps + 3), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 144, 30), 2)

        jpeg = cv2.imencode('.jpg', frame)[1].tostring()
        '''print(jpeg)
        j = jpeg.tobytes()
        print(j)'''
        return jpeg


def gen(camera):
    count = 1
    while True:
        count += 1
        frame_1 = VideoCamera().get_frame(count)
        yield (b'--frame_1\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_1 + b'\r\n\r\n')



# ___________________________________AUTH_UPDATE__________________________________

def indexscreen(request):
    global myauth
    try:
        if myauth:
            template = "screens.html"
            return render(request, template)
        else:
            template = "auth.html"
            return render(request, template)
    except HttpResponseServerError:
        print("aborted")


def auth(request):
    global myauth
    print('-------------------------------------------')
    print(request)
    req = str(request).split('/')
    name = req[2]
    pas = req[3]
    pas = pas[:-2]
    print(str(req))
    print('-------------------------------------------')
    print(name)
    print(pas)
    if name == 'a' and pas == '1':
        myauth = True
        print('auth succesful')
        return HttpResponseRedirect('/stream/screen')


def del_auth(request):
    global myauth
    myauth = False
    return redirect('/stream/screen')


@gzip.gzip_page
def dynamic_stream(request, num=0, stream_path="2.mp4"):
    stream_path = 'add your camera stream here that can rtsp or http'
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

