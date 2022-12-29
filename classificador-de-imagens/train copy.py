from Modelos.brain import Brain
from Modelos.inteligencia import Classificador
from utils.excecoes import ModeloNaoAplicado
import tensorflow as tf
import argparse
from utils.utils import save_model
VGG16        = Brain.VGG16       # Carrega a rede VGG16
RESNET50V2   = Brain.RESNET50V2  # Carrega o modelo ResNet50V2
MOBILENET    = Brain.MOBILENET   # Carrega o modelo MobileNet
MOBILENETV2  = Brain.MOBILENETV2 # Carrega o modelo MobileNetV2
def model_select():
    if opt.model == 'vgg16':
        return Classificador.VGG16
    elif opt.model == 'resnet50v2':
        return Classificador.RESNET50V2
    elif opt.model == 'mobilenet':
        return Classificador.MOBILENET
    elif opt.model == 'mobilenetv2':
        return Classificador.MOBILENETV2
    else:
        raise ModeloNaoAplicado()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16', help='Modelo base pré-treinado, os modelo implementados são: vgg16, resnet50v2, mobilenet e mobilenetv2')
    parser.add_argument('--lr', default= 0.0001, help='Taxa de aprendizagem do modelo')
    parser.add_argument('--train_epochs', type = int, default= 10, help='numero de épocas de treinamento')
    parser.add_argument('--fine_epochs', type = int, default= 10, help='numero de épocas de ajuste fino')
    parser.add_argument('--data', type=str, default='data', help='diretório com as pastas de treinamento e validação')
    parser.add_argument('--name', type=str, default='nei', help='salvamento do modelo em arquivo json')
    opt = parser.parse_args()
    model = Classificador(pretrained_model = model_select(),diretory = opt.data)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr))
    model.train(epochs = opt.train_epochs)
    model.fine_tunning(epochs = opt.fine_epochs)
    model.save_model(caminho = opt.name)