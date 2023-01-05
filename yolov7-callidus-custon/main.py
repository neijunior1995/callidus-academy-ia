import torch
import cv2
from app.interfaces.yolov7_model import yolov7_model
from app.cameo.cameo import Cameo
from registro_de_face.modelos.inteligencia import Classificador


if __name__ == '__main__':
	model = yolov7_model('app/weigths/face_detector.pt',conf_thres = 0.4,iou_thres = 0.40)
	classificador = Classificador(diretory = None)
	classificador.load_model("registro_de_face/weigths/classificador")
	cameo = Cameo(model)
	cameo.run_object(classificador)
	#cameo.run()
