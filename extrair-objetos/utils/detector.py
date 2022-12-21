import cv2

class Detector():
    def __init__(self, data_detec, scale_factor, features):
        self.scale_factor = scale_factor
        self.features = features
        self.detec = cv2.CascadeClassifier(data_detec)
    def detectar(self, caminho):
        self.imagem = cv2.imread(caminho)
        imagem_gray = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
        self.faces = self.detec.detectMultiScale(imagem_gray, self.scale_factor, self.features)
    def salvar(self,output, nome):
        face_imagem = 0
        for (x,y,w,h) in self.faces:
            face_imagem += 1
            imagem_roi = self.imagem[y:y+h, x:x+w]
            cv2.imwrite(output + nome + str(face_imagem) + ".png", imagem_roi)