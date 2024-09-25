import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
import numpy as np
import copy
from time import sleep
from IPython.display import clear_output
import chess
import re

class Triqui:

    def __init__(self):
        self.estado_inicial = np.matrix([[0]*3]*3)

    def pintar_estado(self, estado):
        # Dibuja el tablero correspondiente al estado
        # Input: estado, que es una 3-lista de 3-listas
        fig, axes = plt.subplots()

        # Dibujo el tablero
        step = 1./3
        offset = 0.001
        tangulos = []

        # Borde del tablero
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                          facecolor='cornsilk',\
                                         edgecolor='black',\
                                         linewidth=2))

        # Creo las líneas del tablero
        for j in range(3):
            locacion = j * step
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, locacion), 1, 0.008],\
                    facecolor='black'))
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(locacion, 0), 0.008, 1],\
                    facecolor='black'))

        for t in tangulos:
            axes.add_patch(t)

        # Cargando imagen de O
        arr_img_O = plt.imread("./imagenes/Triqui/O.png", format='png')
        image_O = OffsetImage(arr_img_O, zoom=0.14)
        image_O.image.axes = axes

        # Cargando imagen de X
        arr_img_X = plt.imread("./imagenes/Triqui/X.png", format='png')
        image_X = OffsetImage(arr_img_X, zoom=0.14)
        image_X.image.axes = axes

        offsetX = 0.15
        offsetY = 0.15

        # ASUMO QUE LAS O SE REPRESENTAN CON 1 EN LA MATRIZ
        # Y QUE LAS X SE REPRESENTAN CON 2
        for i in range(3):
            for j in range(3):
                if estado[j, i] == 1:
                    # print("O en (" + str(i) + ", " + str(j) + ")")
                    Y = j
                    X = i
                    # print("(" + str(X) + ", " + str(Y) + ")")
                    ab = AnnotationBbox(
                        image_O,
                        [(X*step) + offsetX, (Y*step) + offsetY],
                        frameon=False)
                    axes.add_artist(ab)
                if estado[j, i] == 2:
                    # print("X en (" + str(i) + ", " + str(j) + ")")
                    Y = j
                    X = i
                    # print("(" + str(X) + ", " + str(Y) + ")")
                    ab = AnnotationBbox(
                        image_X,
                        [(X*step) + offsetX, (Y*step) + offsetY],
                        frameon=False)
                    axes.add_artist(ab)

        axes.axis('off')
        return axes

    def player(self, estado):
        # Devuelve el número del jugador a quien corresponde el turno
        # 1 para las O
        # 2 para las X
        num_Os = np.count_nonzero(estado==1)
        num_Xs = np.count_nonzero(estado==2)
        # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
        if num_Os < num_Xs:
            return 1
        else:
            return 2

    def acciones(self, estado):
        # Devuelve una lista de parejas que representan las casillas vacías
        # Input: estado, que es una np.matrix(3x3)
        # Output: lista de índices (x,y)
        indices = []
        if np.count_nonzero(estado==0)>0:
            for x in range(3):
                for y in range(3):
                    if estado[y, x] == 0:
                        indices.append((x, y))

        return indices

    def resultado(self, estado, indice):
        # Devuelve el tablero incluyendo una O o X en el indice,
        # dependiendo del jugador que tiene el turno
        # Input: estado, que es una np.matrix(3x3)
        #        indice, de la forma (x,y)
        # Output: estado, que es una np.matrix(3x3)

        s = copy.deepcopy(estado)
        x = indice[0]
        y = indice[1]
        s[y, x] = self.player(estado)

        return s

    def es_terminal(self, estado):
        # Devuelve True/False dependiendo si el juego se acabó
        # Input: estado, que es una np.matrix(3x3)
        # Output: objetivo, True/False
        # print("Determinando si no hay casillas vacías...")
        if np.count_nonzero(estado==0)==0:
            return True
        else:
            # print("Buscando triqui horizontal...")
            for y in range(3):
                num_Os = np.count_nonzero(estado[y,:]==1)
                num_Xs = np.count_nonzero(estado[y,:]==2)
                # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
                if (num_Os==3) or (num_Xs==3):
                    return True
            # print("Buscando triqui vertical...")
            for x in range(3):
                num_Os = np.count_nonzero(estado[:,x]==1)
                num_Xs = np.count_nonzero(estado[:,x]==2)
                # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
                if (num_Os==3) or (num_Xs==3):
                    return True
            # print("Buscando triqui diagonal...")
            if (estado[0,0]==1) and (estado[1,1]==1) and (estado[2,2]==1):
                return True
            elif (estado[0,0]==2) and (estado[1,1]==2) and (estado[2,2]==2):
                return True
            # print("Buscando triqui transversal...")
            if (estado[2,0]==1) and (estado[1,1]==1) and (estado[0,2]==1):
                return True
            elif (estado[2,0]==2) and (estado[1,1]==2) and (estado[0,2]==2):
                return True
        return False

    def utilidad(self, estado, jugador):
		# Devuelve la utilidad del estado donde termina el juego
		# Input: estado, que es una np.matrix(3x3)
		# Output: utilidad, que es un valor -1, 0, 1
		# print("Buscando triqui horizontal
        for y in range(3):
            num_Os = np.count_nonzero(estado[y,:]==1)
            num_Xs = np.count_nonzero(estado[y,:]==2)
            # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
            if (num_Os==3):
                return -1
            elif (num_Xs==3):
                return 1
            # print("Buscando triqui vertical...")
            for x in range(3):
                num_Os = np.count_nonzero(estado[:,x]==1)
                num_Xs = np.count_nonzero(estado[:,x]==2)
                # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
                if (num_Os==3):
                    return -1
                elif (num_Xs==3):
                    return 1
            # print("Buscando triqui diagonal...")
            if (estado[0,0]==1) and (estado[1,1]==1) and (estado[2,2]==1):
                return -1
            elif (estado[0,0]==2) and (estado[1,1]==2) and (estado[2,2]==2):
                return 1
            # print("Buscando triqui transversal...")
            if (estado[2,0]==1) and (estado[1,1]==1) and (estado[0,2]==1):
                return -1
            elif (estado[2,0]==2) and (estado[1,1]==2) and (estado[0,2]==2):
                return 1
            # Determina si hay empate
            if np.count_nonzero(estado==0)==0:
                return 0
        return None

class ReyReinaRey:

    '''
    Usa la librería python-chess
    Documentación en https://python-chess.readthedocs.io/en/latest/index.html
    '''

    def __init__(self, jugador='negras', max_lim=2, tablero_inicial=1):
        self.max_lim = max_lim
        if jugador == 'blancas':
            pl = ' w'
        elif jugador == 'negras':
            pl = ' b'
        else:
            raise NameError('¡Jugador incorrecto! Debe ser \'blancas\' o \'negras\'.' )
        
        if tablero_inicial == "random":
            self.estado_inicial = self.random_board(pl)
        else:
            dict_tableros = {3:chess.Board("8/8/8/3k4/8/8/6K1/7R" + pl),\
                             1:chess.Board("2R5/8/8/8/8/8/k1K5/8" + pl),\
                             4:chess.Board("8/8/8/4k3/8/8/5K2/5Q2" + pl),\
                             5:chess.Board("8/8/8/4k3/8/5Q2/5K2/8" + pl),\
                             2:chess.Board("8/8/8/8/8/8/1k1K4/2R5" + pl)}
            self.estado_inicial = dict_tableros[tablero_inicial]
    
    def random_board(self, pl):
        
        piece_dict = {1:"k", 2:"K", 3:"Q"}

        pieces = [1, 2, 3]
        def parse_board(array:np.array):
            board = ""
            for row in array:
                empty_count = 0
                temp = ""
                for square in row:
                    if square == 0:
                        empty_count += 1
                    if square in pieces:
                        s = str(empty_count) if empty_count != 0 else ""
                        temp += s + piece_dict[square]
                        empty_count = 0
                s = str(empty_count) if empty_count != 0 else ""
                temp += s + "/"
                empty_count = 0
                board += temp
            board = board[:-1] + pl
            print(board)
            return chess.Board(board)
        
        board = np.zeros((8, 8), dtype=int)
        positions = [str(x)+str(y) for y in range(8) for x in range(8)]
        
        # Verificamos que la posición inicial no sea jaque
        cond = False
        while not cond:
            cond = True
            k, K, Q = np.random.choice(positions, 3, replace=False)

            kx, ky = int(k[0]), int(k[1])
            Kx, Ky = int(K[0]), int(K[1])
            Qx, Qy = int(Q[0]), int(Q[1])

            # Verificamos que los reyes no estén juntos
            if abs(kx - Kx) == 1 or abs(ky - Ky) == 1:
                 cond = False

            # Verificamos que la reina no ataque al rey
            if (kx == Qx) or (ky == Qy) or ((kx-ky) == (Qx-Qy)) or ((kx+ky) == (Qx+Qy)):
                 cond = False

        board[kx, ky] = 1
        board[Kx, Ky] = 2
        board[Qx, Qy] = 3
        
        return parse_board(board)
        
    
    
    
    def pintar_estado(self, board):
        # Dibuja el tablero correspondiente al estado
        # Input: estado
        return board

    def player(self, board):
        # Devuelve el jugador a quien corresponde el turno
        if board.turn:
            return 'blancas'
        else:
            return 'negras'

    def acciones(self, board):
        # Devuelve una lista de acciones legales en el tablero
        # Input: estado
        # Output: lista de jugadas en notación algebráica estándar (SAN)
        return list(board.legal_moves)

    def jugada_manual(self, board, accion):
        if board.parse_san(accion) in board.legal_moves:
            board_copy = copy.deepcopy(board)
            board_copy.push_san(accion)
            return board_copy
        else:
            raise NameError('Formato de acción incorrecta. Debe estar en notación algebráica estándar.')

    def resultado(self, board, accion):
        # Devuelve el tablero resultado de la jugada,
        # dependiendo del jugador que tiene el turno
        # Input: estado
        #        accion
        # Output: estado
        board_copy = copy.deepcopy(board)
#        print(board_copy)
        board_copy.push(accion)
        return board_copy

    def es_terminal(self, board):
        # Devuelve True/False dependiendo si el juego se acabó
        # Input: estado, que es una np.matrix(3x3)
        # Output: objetivo, True/False
        # print("Determinando si no hay casillas vacías...")
        if board.outcome() is not None:
            return True
        else:
            return False

    def utilidad(self, board, jugador):
        # Devuelve la utilidad del estado donde termina el juego
        # Input: estado, que es una np.matrix(3x3)
        # Output: utilidad, que es un valor -1, 0, 1
        if board.outcome() is not None:
            fin = str(board.outcome().termination)
            if 'CHECK' in fin:
                if board.turn:
                    return -1000
                else:
                    return 1000
            else:
                return 0
        else:
            return None

    def casilla_pieza(self, board, pieza):
        tablero = str(board).split('\n')
        fila = [i for i in range(len(tablero)) if pieza in tablero[i]][0]
        columna = tablero[fila].replace(' ', '').find(pieza)
        return (fila, columna)
    
    def evaluar(self, board, jugador):
        if self.es_terminal(board):
            return self.utilidad(board, jugador)
        else:
            pesos = [1, 4, 16, 16, 4, 1]
            valores = [self.rey_borde(board),\
                       self.material(board),\
                       self.oposicion(board),\
                       self.salto_de_caballo(board),\
                       self.reducir_fila_y_columna(board),\
                       self.revisar_mate(board)\
                      ]
            return np.dot(pesos, valores)
            return np.dot(pesos, valores)

    def rey_borde(self, board):
        # Contamos rey negro en borde
        fila, columna = self.casilla_pieza(board, 'k')
        rey_negro_fila = (4 - fila if fila < 4 else (fila % 4) + 1) - 3 
        rey_negro_columna = (4 - columna if columna < 4 else (columna % 4)) - 3
        rincon = max(rey_negro_fila, rey_negro_columna)
        return 10 * np.exp(rincon)

    def material(self, board):
        # Contamos material
        piezas = re.findall(r"[\w]+", str(board))
        dict_material = {'K':90, 'R':5, 'k':-90, "Q":9}
        piezas = [dict_material[p] for p in piezas]
        material = np.sum(piezas)
        return material
   
    def oposicion(self, board):
        # Contamos oposición
        fila_rey_blanco, columna_rey_blanco = self.casilla_pieza(board, 'K')
        fila_rey_negro, columna_rey_negro = self.casilla_pieza(board, 'k')
        distancia_fila = np.abs(fila_rey_blanco - fila_rey_negro)
        distancia_columna = np.abs(columna_rey_blanco - columna_rey_negro)
        if distancia_fila == 0:
            oposicion = distancia_columna - 2
        elif distancia_columna == 0:
            oposicion = distancia_fila - 2
        else:
            oposicion = distancia_fila + distancia_columna
        return 10 * np.exp(-oposicion)
    
    def salto_de_caballo(self, board):
        row_k, col_k = self.casilla_pieza(board, 'k')
        row_Q, col_Q = self.casilla_pieza(board, 'Q')
        
        r = abs(row_k - row_Q)
        c = abs(col_k - col_Q)
        
        return 2 * int((r == 1 and c == 2) or (r == 2 and c == 1)) - 1
    
    
    def reducir_fila_y_columna(self, board):
        base = 1.75
        row_k, col_k = self.casilla_pieza(board, 'k')
        try:
            row_Q, col_Q = self.casilla_pieza(board, 'Q')
        except:
            return 0
        
        suma = 0
        # Reducir fila:
        if row_k < row_Q:
            suma += base**(8 - row_Q)
        elif row_k > row_Q:
            suma += base**(row_Q)
        else:
            suma += base

        # Reducir fila:
        if col_k < col_Q:
            suma += base**(8 - col_Q)
        elif col_k > col_Q:
            suma += base**(col_Q)
        else:
            suma += base
            
        return suma
        
    def revisar_mate(self, board):
        if board.is_checkmate():
            return np.inf
        return 0

    def num_acc_contrincante(self, board):
        n = len(self.acciones(board))
        return 100 * np.exp(-n)
    
    def is_cutoff(self, board, d):
        if self.es_terminal(board):
            return True
        elif d >= self.max_lim:
            return True
        else:
            return False
        
    def H_minimax_alfa_beta(self, board, d, alfa, beta):
        jugador = self.player(board)
        if self.is_cutoff(board, d):
            return self.evaluar(board, jugador) - 0.1 * d, None
        elif jugador == 'blancas':
            v = -np.infty
            for a in self.acciones(board):
                board_resultado = self.resultado(board, a)
                v2, a2 = self.H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if v2 > v:
                    v = v2
                    accion = a
                    alfa = max(alfa, v)
                if v >= beta:
                    return v, accion
            return v, accion
        elif jugador == 'negras':
            v = np.infty
            for a in self.acciones(board):
                board_resultado = self.resultado(board, a)
                v2, a2 = self.H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if v2 < v:
                    v = v2
                    accion = a
                    beta = min(beta, v)
                if v <= alfa:
                    return v, accion
            return v, accion
        else:
            raise NameError("Oops!")
            
            
class ReyTorreRey:

    '''
    Usa la librería python-chess
    Documentación en https://python-chess.readthedocs.io/en/latest/index.html
    '''

    def __init__(self, jugador='negras', max_lim=2, tablero_inicial=1):
        self.max_lim = max_lim
        if jugador == 'blancas':
            pl = ' w'
        elif jugador == 'negras':
            pl = ' b'
        else:
            raise NameError('¡Jugador incorrecto! Debe ser \'blancas\' o \'negras\'.' )
        
        if tablero_inicial == "random":
            self.estado_inicial = self.random_board(pl)
        else:
            dict_tableros = {3:chess.Board("8/8/8/3k4/8/8/6K1/7R" + pl),\
                             1:chess.Board("2R5/8/8/8/8/8/k1K5/8" + pl),\
                             4:chess.Board("8/8/8/4k3/8/8/5K2/5Q2" + pl),\
                             5:chess.Board("8/8/8/4k3/8/5Q2/5K2/8" + pl),\
                             2:chess.Board("8/8/8/8/8/8/1k1K4/2R5" + pl)}
            self.estado_inicial = dict_tableros[tablero_inicial]
    
    def random_board(self, pl):
        
        piece_dict = {1:"k", 2:"K", 3:"Q"}

        pieces = [1, 2, 3]
        def parse_board(array:np.array):
            board = ""
            for row in array:
                empty_count = 0
                temp = ""
                for square in row:
                    if square == 0:
                        empty_count += 1
                    if square in pieces:
                        s = str(empty_count) if empty_count != 0 else ""
                        temp += s + piece_dict[square]
                        empty_count = 0
                s = str(empty_count) if empty_count != 0 else ""
                temp += s + "/"
                empty_count = 0
                board += temp
            board = board[:-1] + pl
            print(board)
            return chess.Board(board)
        
        board = np.zeros((8, 8), dtype=int)
        positions = [str(x)+str(y) for y in range(8) for x in range(8)]
        
        # Verificamos que la posición inicial no sea jaque
        cond = False
        while not cond:
            cond = True
            k, K, Q = np.random.choice(positions, 3, replace=False)

            kx, ky = int(k[0]), int(k[1])
            Kx, Ky = int(K[0]), int(K[1])
            Qx, Qy = int(Q[0]), int(Q[1])

            # Verificamos que los reyes no estén juntos
            if abs(kx - Kx) == 1 or abs(ky - Ky) == 1:
                 cond = False

            # Verificamos que la reina no ataque al rey
            if (kx == Qx) or (ky == Qy) or ((kx-ky) == (Qx-Qy)) or ((kx+ky) == (Qx+Qy)):
                 cond = False

        board[kx, ky] = 1
        board[Kx, Ky] = 2
        board[Qx, Qy] = 3
        
        return parse_board(board)

    def pintar_estado(self, board):
        # Dibuja el tablero correspondiente al estado
        # Input: estado
        return board

    def player(self, board):
        # Devuelve el jugador a quien corresponde el turno
        if board.turn:
            return 'blancas'
        else:
            return 'negras'

    def acciones(self, board):
        # Devuelve una lista de acciones legales en el tablero
        # Input: estado
        # Output: lista de jugadas en notación algebráica estándar (SAN)
        return list(board.legal_moves)

    def jugada_manual(self, board, accion):
        if board.parse_san(accion) in board.legal_moves:
            board_copy = copy.deepcopy(board)
            board_copy.push_san(accion)
            return board_copy
        else:
            raise NameError('Formato de acción incorrecta. Debe estar en notación algebráica estándar.')

    def resultado(self, board, accion):
        # Devuelve el tablero resultado de la jugada,
        # dependiendo del jugador que tiene el turno
        # Input: estado
        #        accion
        # Output: estado
        board_copy = copy.deepcopy(board)
#        print(board_copy)
        board_copy.push(accion)
        return board_copy

    def es_terminal(self, board):
        # Devuelve True/False dependiendo si el juego se acabó
        # Input: estado, que es una np.matrix(3x3)
        # Output: objetivo, True/False
        # print("Determinando si no hay casillas vacías...")
        if board.outcome() is not None:
            return True
        else:
            return False

    def utilidad(self, board, jugador):
        # Devuelve la utilidad del estado donde termina el juego
        # Input: estado, que es una np.matrix(3x3)
        # Output: utilidad, que es un valor -1, 0, 1
        if board.outcome() is not None:
            fin = str(board.outcome().termination)
            if 'CHECK' in fin:
                if board.turn:
                    return -1000
                else:
                    return 1000
            else:
                return 0
        else:
            return None

    def casilla_pieza(self, board, pieza):
        tablero = str(board).split('\n')
        fila = [i for i in range(len(tablero)) if pieza in tablero[i]][0]
        columna = tablero[fila].replace(' ', '').find(pieza)
        return (fila, columna)
    
    def evaluar(self, board, jugador):
        if self.es_terminal(board):
            return self.utilidad(board, jugador)
        else:
            pesos = [1, 1, 1]
            valores = [self.rey_borde(board),\
                       self.material(board),\
                       self.oposicion(board),\
                      ]
            return np.dot(pesos, valores)

    def rey_borde(self, board):
        # Contamos rey negro en borde
        fila, columna = self.casilla_pieza(board, 'k')
        rey_negro_fila = (4 - fila if fila < 4 else (fila % 4) + 1) - 3 
        rey_negro_columna = (4 - columna if columna < 4 else (columna % 4)) - 3
        rincon = max(rey_negro_fila, rey_negro_columna)
        return 10 * np.exp(rincon)

    def material(self, board):
        # Contamos material
        piezas = re.findall(r"[\w]+", str(board))
        dict_material = {'K':90, 'R':5, 'k':-90, "Q":9}
        piezas = [dict_material[p] for p in piezas]
        material = np.sum(piezas)
        return material
   
    def oposicion(self, board):
        # Contamos oposición
        fila_rey_blanco, columna_rey_blanco = self.casilla_pieza(board, 'K')
        fila_rey_negro, columna_rey_negro = self.casilla_pieza(board, 'k')
        distancia_fila = np.abs(fila_rey_blanco - fila_rey_negro)
        distancia_columna = np.abs(columna_rey_blanco - columna_rey_negro)
        if distancia_fila == 0:
            oposicion = distancia_columna - 2
        elif distancia_columna == 0:
            oposicion = distancia_fila - 2
        else:
            oposicion = distancia_fila + distancia_columna
        return 10 * np.exp(-oposicion)
    
    def num_acc_contrincante(self, board):
        n = len(self.acciones(board))
        return 100 * np.exp(-n)
    
    def is_cutoff(self, board, d):
        if self.es_terminal(board):
            return True
        elif d >= self.max_lim:
            return True
        else:
            return False
        
    def H_minimax_alfa_beta(self, board, d, alfa, beta):
        jugador = self.player(board)
        if self.is_cutoff(board, d):
            return self.evaluar(board, jugador) - 0.1 * d, None
        elif jugador == 'blancas':
            v = -np.infty
            for a in self.acciones(board):
                board_resultado = self.resultado(board, a)
                v2, a2 = self.H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if v2 > v:
                    v = v2
                    accion = a
                    alfa = max(alfa, v)
                if v >= beta:
                    return v, accion
            return v, accion
        elif jugador == 'negras':
            v = np.infty
            for a in self.acciones(board):
                board_resultado = self.resultado(board, a)
                v2, a2 = self.H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if v2 < v:
                    v = v2
                    accion = a
                    beta = min(beta, v)
                if v <= alfa:
                    return v, accion
            return v, accion
        else:
            raise NameError("Oops!")  