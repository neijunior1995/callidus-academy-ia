from utils.detector import Detector
from os import listdir, path, makedirs
from os.path import isfile, join
import argparse
if __name__ == '__main__':
    """Implementação dos métodos utilizados para detectar os objetos"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/', help='Diretório com as imagens a serem carregadas')
    parser.add_argument('--scale-factor', type=str, default=1.2, help='Fator de escala para detecção de objetos')
    parser.add_argument('--features', type=str, default=6, help='Numero mínimo de características para detectar o objeto')
    parser.add_argument('--detector', type=str, default='classificadores/haarcascade_frontalface_default.xml', help='caminho onde deve ser buscado o detector haar cascade')
    parser.add_argument('--output', type=str, default='out/', help='Diretório onde as imagens devem ser salvas')
    opt = parser.parse_args()

    lista_arquivos = [f for f in listdir(opt.data) if isfile(join(opt.data, f))]
    detector = Detector(opt.detector, opt.scale_factor, int(opt.features))
    if not path.exists(opt.output):
        makedirs(opt.output)
    for arq in lista_arquivos:
        detector.detectar(opt.data + arq)
        nome  = arq.split('.')[0]
        print(opt.data + arq+ " possui: "+str(len(detector.faces))+" faces detectadas")
        detector.salvar(opt.output,nome)