from Modelos.brain import Brain
from Modelos.inteligencia import Classificador

if __name__ == '__main__':
    model = Classificador(diretory = None)
    model.load_model('classificacador-de-faces\classificador')
    print(model.predict_file('teste_nei.jpg'))