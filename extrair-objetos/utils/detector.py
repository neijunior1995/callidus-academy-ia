import cv2

class Detector():
    """Esta classe é utilizada para aplicar o classificador que será utilizado para extrair objetos das imagens"""
    def __init__(self, data_detec, scale_factor, features):
        """O método init é utilizado para carregar o modelo do detector utilizado para detecão de objetos"""
        self.scale_factor = scale_factor # Fator de escala para deteção de objetos hasscascade
        self.features = features         # Números de caracterísiticas
        self.detec = cv2.CascadeClassifier(data_detec) # Detector utilizado
    def detectar(self, caminho):
        """ Método utilizado para implementar a detecção do objeto"""
        self.imagem = cv2.imread(caminho) # Imagem carregada
        imagem_gray = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY) # Imagem carrega em escala de cinza
        self.faces = self.detec.detectMultiScale(imagem_gray, self.scale_factor, self.features) # Array com os dados detectados
    def salvar(self,output, nome):
        """ Método utilizado para salvar os dados"""
        face_imagem = 0
        for (x,y,w,h) in self.faces:
            face_imagem += 1
            imagem_roi = self.imagem[y:y+h, x:x+w]
            cv2.imwrite(output + nome + str(face_imagem) + ".png", imagem_roi)