import torch
import cv2


def labelDraw(image,detect,class_names, color_rec = (255, 255, 0), rec_width = 2, font = cv2.FONT_HERSHEY_COMPLEX_SMALL, width_font = 0.5, color_font = (255,0,0)):
	for det in detect:
		x,y,x2,y2 = det[:4]
		cv2.rectangle(image, (int(x) , int(y)), (int(x2) , int(y2)), color_rec, rec_width)
		label = "Class {0}, conf {conf:.2f}".format(class_names[int(det[5])],conf = det[4])
		cv2.putText(image,label,(int(x),int(y)),font,width_font,color_font)

def labelDrawOneDetect(image,det,label_classificado, color_rec = (255, 255, 0), rec_width = 2, font = cv2.FONT_HERSHEY_COMPLEX_SMALL, width_font = 1, color_font = (255,0,0)):
	x,y,x2,y2 = det[:4]
	cv2.rectangle(image, (int(x) , int(y)), (int(x2) , int(y2)), color_rec, rec_width)
	label = "Nome: {0}".format(label_classificado)
	cv2.putText(image,label,(int(x),int(y)),font,width_font,color_font)

def roi(imagem, det):
	x,y,x2,y2 = det[:4]
	return[imagem[y:y2,x:x2]]


def onImage(image,model):
	image_display = image.copy()
	image_display = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
	detect = model.processFrame(image_display)
	
	return detect
def load_image(self,classificado,output_size):
    """Carrega uma imagem"""
    imagem = cv2.resize(imagem,output_size)
    imagem = imagem.reshape(1,output_size[0],output_size[1],output_size[2])
    return imagem
def batch_image(dataset,output_size):
	return dataset.reshape(-1,output_size[0],output_size[1],output_size[2])
