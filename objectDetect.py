import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classNames = ['pessoa', 'bicicleta', 'carro', 'motocicleta', 'avião',
               'ônibus', 'trem', 'caminhão', 'barco', 'semáforo', 'hidrante',
                 'placa de rua', 'placa de pare', 'parquímetro', 'banco', 'pássaro',
                   'gato', 'cachorro', 'cavalo', 'ovelha', 'vaca', 'elefante', 'urso',
                     'zebra', 'girafa', 'chapéu', 'mochila', 'guarda-chuva', 'sapato', 
                     'óculos', 'bolsa', 'gravata', 'mala', 'frisbee', 'esquis', 'snowboard',
                       'bola de esportes', 'pipa', 'taco de beisebol', 'luva de beisebol', 'skate',
                         'prancha de surfe', 'raquete de tênis', 'garrafa', 'prato', 'taça de vinho', 
                         'copo', 'garfo', 'faca', 'colher', 'tigela', 'banana', 'maçã', 'sanduíche',
                           'laranja', 'brócolis', 'cenoura', 'cachorro-quente', 'pizza', 'donut',
                             'bolo', 'cadeira', 'sofá', 'planta em vaso', 'cama', 'espelho',
                               'mesa de jantar', 'janela', 'escrivaninha', 'vaso sanitário',
                                 'porta', 'tv', 'laptop', 'mouse', 'controle remoto', 'teclado', 
                                 'celular', 'micro-ondas', 'forno', 'torradeira', 'pia', 'geladeira',
                                   'liquidificador', 'livro', 'relógio', 'vaso', 'tesoura',
                                     'ursinho de pelúcia', 'secador de cabelo', 'escova de dente',
                                       'escova de cabelo']

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputscale(1.0/127.5)
net.setImput

#https://www.youtube.com/watch?v=Yw4IrVeylvY&list=PL_36D3ID4tO-xgGzgfFwMvPqyt0U83hO8&index=8
#continuar min 8:44