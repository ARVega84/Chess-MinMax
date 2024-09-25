# From Python
import numpy as np
from random import choice
import pandas as pd
import seaborn as sns
from time import sleep
import matplotlib.pyplot as plt
import warnings
# From notebook
from juegos import Triqui, ReyTorreRey
from tiempos import compara_funciones
from copy import deepcopy

# Búsqueda con profundidad limitada
roo = ReyTorreRey(max_lim = 2, tablero_inicial=3)
s = roo.estado_inicial
print('//////////////')
print(s)
valores = [roo.rey_borde(s),\
            roo.material(s),\
            roo.oposicion(s),\
                ]
print(valores)

acciones_negro = ['Kb3', 'Kb4', 'Kb5', 'Kb6', 'Kb5', 'Kb6', \
                  'Kb7',\
                  'Kb6', 'Ka5', 'Ka4', 'Ka3', 'Ka2', 'Ka3', \
                  'Kb8', 'Kc8',\
                  'Kb7', 'Ka6', 'Ka5']
acciones_blanco = ['Kd3', 'Kd4']
for i, accion in enumerate(acciones_negro):
    print('Negro:', accion)
    s = roo.jugada_manual(s, accion)
    print('//////////////')
    print(s)
    valores = [roo.rey_borde(s),\
                roo.material(s),\
                roo.oposicion(s),\
                    ]
    print(valores)
    # accion_ = acciones_blanco[i]
    # print('Blanco:', accion_)
    # s = roo.jugada_manual(s, accion_)
    # print('//////////////')
    # print(s)
    # valores = [roo.rey_borde(s),\
    #             roo.material(s),\
    #             roo.oposicion(s),\
    #                 ]
    # print(valores)
    if not roo.es_terminal(s):
        acciones_blanco = roo.acciones(s)
        # print('Acciones posibles blanco:', acciones_blanco)
        for a in acciones_blanco:
            board_resultado = roo.resultado(s, a)
            v2, a2 = roo.H_minimax_alfa_beta(board_resultado, 1, -np.infty, np.infty)
            # print(a, v2, a2)
        v, a = roo.H_minimax_alfa_beta(s, 0, -np.infty, np.infty)
        print('Blanco:', a)
        s = roo.resultado(s, a)
        sleep(1)
        print('//////////////')
        print(s)
        valores = [roo.rey_borde(s),\
                    roo.material(s),\
                    roo.oposicion(s),\
                      ]
        print(valores)
        if roo.es_terminal(s):
            print('Juego terminado. ¡Ganan las blancas!')
            break
    else:
        jugador = roo.player(s)
        if roo.utilidad(s, jugador)==0:
            print('Juego terminado. ¡Tablas!')
        else:
            print('Juego terminado. ¡Ganan las negras!')
        break