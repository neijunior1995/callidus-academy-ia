from Modelos.brain import Brain
from Modelos.inteligencia import Classificador
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='classificador-de-faces/clasifica-faces', help='Parâmetro utilizado para identificar o caminho do modelo carregado')
    parser.add_argument('--data', type=str, default='teste.jpg', help='caminho da pasta testada')
    opt = parser.parse_args()

    model = Classificador(diretory = None)
    model.load_model(opt.model)
    print(model.predict_file(opt.data))