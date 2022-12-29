from Modelos.brain import Brain
from Modelos.inteligencia import Classificador
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    model = Classificador(diretory = None)
    model.load_model('classificacador-de-faces\classificador')
    image = model.model.load_image('teste_nei.jpg')
    print(image.shape)
    image = np.append(image,model.model.load_image('teste_leticia.jpg'))
    image = image.reshape(-1,160,160,3)
    print(image.shape)
    print(model.predict(image))