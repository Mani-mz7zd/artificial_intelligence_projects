#
# raichu.py : Play the game of Raichu
#
# Pothapragada Venkata SG Krishna Srikar: vpothapr, Manikanta Kodandapani Naidu: k11, G Vivek Reddy: gvi

import sys
import numpy as np
import copy

def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def Print_board(board):
    string = ''
    for row in range(len(board)) :
        for col in range(len(board[0])):
            string = string + board[row][col]
    
    return string

def Locations(Board2D,player):
    n = len(Board2D)
    Loc = [[],[],[]]
    
    if player == 'w':
        for row in range(n):
            for col in range(n):
                if Board2D[row][col] in "w":
                    Loc[0].append((row,col))
                elif Board2D[row][col] in "W":
                    Loc[1].append((row,col))
                elif Board2D[row][col] in "@":
                    Loc[2].append((row,col))

    else:
        for row in range(n):
            for col in range(n):
                if Board2D[row][col] in "b":
                    Loc[0].append((row,col))
                elif Board2D[row][col] in "B":
                    Loc[1].append((row,col))
                elif Board2D[row][col] in "$":
                    Loc[2].append((row,col))
    return Loc,n

def Vertical(Loc_w,Loc_b,n):

    distance_w,distance_b = 0,0
    
    for loc in Loc_w[0:2]:
        for i in loc:
            distance_w = distance_w + (n-i[0])
    
    for loc in Loc_b[0:2]:
        for i in loc:
            distance_b = distance_b + (i[0])
    
    if player == 'w':
        return distance_b - distance_w
    else:
        return distance_w - distance_b
    
    #return 0
     
def evaluate_function(board,player,depth):
    vertical_power = 0
    Loc_w,n = Locations(board, 'w')
    Loc_b,n = Locations(board, 'b')

    if player == 'w':
        score = {'w' :10, 'W':30, '@': 50, 'b': -10, 'B': -30, '$': -50 }
        points = 0
        for i,j in enumerate(['w','W','@']):
            points += len(Loc_w[i])*score[j]
        for i,j in enumerate(['b','B','$']):
            points += len(Loc_b[i])*score[j]  
        
    else:
        score = {'w' :-10, 'W':-30, '@': -50, 'b': 10, 'B': 30, '$': 50 }
        points = 0
        for i,j in enumerate(['w','W','@']):
            points += len(Loc_w[i])*score[j]
        for i,j in enumerate(['b','B','$']):
            points += len(Loc_b[i])*score[j]
    
    if any(Loc_w) and any(Loc_b):
        vertical_power = Vertical(Loc_w,Loc_b,n)
    else:
        if any(Loc_w) != True:
            if player == 'w':
                return -100000 - depth
            else:
                return 100000 + depth
            
        if any(Loc_b) != True:
            if player == 'w':
                return 100000 + depth
            else:
                return -100000 - depth
    
    return points + depth + vertical_power

def check_mate(board,player):

    if player == 'b':
        pawn_locs = [(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i] in "wW@"]
        if len(pawn_locs) == 0:
            return True
    else:
        pawn_locs = [(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i] in "bB$"]
        if len(pawn_locs) == 0:
            return True
    
    return False

def valid_move(a,b,n):
    if 0 <= a < n and 0 <= b < n:
        return True
    
    return False

def Raichu_empty(Initial, Final,Board,move):
    if Initial == Final:
        return False
    
    if move == 'UD':
        Distance = abs(Final[0] - Initial[0])

        if Distance == 1:
                return True
        else:
            if Final[0] > Initial[0]:
                for i in range(Initial[0]+1,Final[0]):
                    if Board[i][Initial[1]] != '.':
                        return False
            else:
                for i in range(Initial[0]-1,Final[0],-1):
                    if Board[i][Initial[1]] != '.':
                        return False
            
            return True
                    
    elif move == 'LR':

        Distance = abs(Final[1] - Initial[1])

        if Distance == 1:
                return True
        
        else:
            if Final[1] > Initial[1]:
                for i in range(Initial[1]+1,Final[1]):
                    if Board[Initial[0]][i] != '.':
                        return False
            else:
                for i in range(Initial[1]-1,Final[1],-1):
                    if Board[Initial[0]][i] != '.':
                        return False

            return True
    
    else:

        Distance1 = abs(Final[1] - Initial[1])
        Distance2 = abs(Final[0] - Initial[0])

        if Distance1 == 1 and Distance2 == 1:
            return True
        
        else:
            if Final[1] > Initial[1]:
                if Final[0] > Initial[0]: #Right-Down
                    range1 = range(Initial[0]+1,Final[0])
                    range2 = range(Initial[1]+1,Final[1])
                    for i,j in zip(range1,range2):
                        if Board[i][j] != '.':
                            return False
                else: #Right-Up
                    range1 = range(Initial[0]-1,Final[0],-1)
                    range2 = range(Initial[1]+1,Final[1])
                    for i,j in zip(range1,range2):
                        if Board[i][j] != '.':
                            return False
            else:
                if Final[0] > Initial[0]: #Left-Down

                    range1 = range(Initial[0]+1,Final[0])
                    range2 = range(Initial[1]-1,Final[1],-1)
                    for i,j in zip(range1,range2):
                        if Board[i][j] != '.':
                            return False
                        
                else: #Left-Up
                    range1 = range(Initial[0]-1,Final[0],-1)
                    range2 = range(Initial[1]-1,Final[1],-1)
                    for i,j in zip(range1,range2):
                        if Board[i][j] != '.':
                            return False

            return True

def Raichu_moves(Board,Location,player,n):
    moves = list()

    if player == 'w':
        opp = 'bB$'
        cur = '@'
    else:
        opp = 'wW@'
        cur = '$'

    for Loc in Location:
        for k in range(1-n,n):
            #Up and Down
            if valid_move(Loc[0]+k,Loc[1],n):
                if Board[Loc[0]+k][Loc[1]] == '.' and Raichu_empty(Loc,(Loc[0]+k,Loc[1]),Board,'UD'):
                    new_Board = copy.deepcopy(Board)
                    new_Board[Loc[0]+k][Loc[1]] = cur
                    new_Board[Loc[0]][Loc[1]] = '.'
                    moves.append(new_Board)

                elif Board[Loc[0]+k][Loc[1]] in opp and Raichu_empty(Loc,(Loc[0]+k,Loc[1]),Board,'UD'):
                    x = -1 if k < 0 else 1
                    if valid_move(Loc[0]+k+x,Loc[1],n) and Board[Loc[0]+k+x][Loc[1]] == '.':
                        new_Board = copy.deepcopy(Board)
                        new_Board[Loc[0]+k+x][Loc[1]] = cur
                        new_Board[Loc[0]+k][Loc[1]] = '.'
                        new_Board[Loc[0]][Loc[1]] = '.'
                        moves.append(new_Board)
            #Right and Left

            if valid_move(Loc[0],Loc[1]+k,n):

                if Board[Loc[0]][Loc[1]+k] == '.' and Raichu_empty(Loc,(Loc[0],Loc[1]+k),Board,'LR'):
                    new_Board = copy.deepcopy(Board)
                    new_Board[Loc[0]][Loc[1]+k] = cur
                    new_Board[Loc[0]][Loc[1]] = '.'
                    moves.append(new_Board)

                elif Board[Loc[0]][Loc[1]+k] in opp and Raichu_empty(Loc,(Loc[0],Loc[1]+k),Board,'LR'):
                    x = -1 if k < 0 else 1
                    if valid_move(Loc[0],Loc[1]+k+x,n) and Board[Loc[0]][Loc[1]+k+x] == '.':
                        new_Board = copy.deepcopy(Board)
                        new_Board[Loc[0]][Loc[1]+k+x] = cur
                        new_Board[Loc[0]][Loc[1]+k] = '.'
                        new_Board[Loc[0]][Loc[1]] = '.'
                        moves.append(new_Board)
    
        for k in range(n):
            
            #Diagonal:
            for t in [-1,1]:
                for y in [-1,1]:
                    if valid_move(Loc[0]+t*k,Loc[1]+y*k,n):
                        if Board[Loc[0]+t*k][Loc[1]+y*k] == '.' and Raichu_empty(Loc,(Loc[0]+t*k,Loc[1]+y*k),Board,'Dia'):
                            
                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]+t*k][Loc[1]+y*k] = cur
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)

                        elif Board[Loc[0]+t*k][Loc[1]+y*k] in opp and Raichu_empty(Loc,(Loc[0]+t*k,Loc[1]+y*k),Board,'Dia'):
                            x = -1 if t*k < 0 else 1
                            z = -1 if y*k < 0 else 1
                            if valid_move(Loc[0]+t*k+x,Loc[1]+y*k+z,n) and Board[Loc[0]+t*k+x][Loc[1]+y*k+z] == '.':
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]+t*k+x][Loc[1]+y*k+z] = cur
                                    new_Board[Loc[0]+t*k][Loc[1]+y*k] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
    
    return moves

def empty(Initial,Final,Board,n,player,move):
    if Initial == Final:
        return False
    if valid_move(Final[0],Final[1],n):
            if move == 'UD':
                Distance = abs(Final[0] - Initial[0])

                if Distance == 1:
                    return True
                
                else:
                    if player == 'w':
                        for i in range(Initial[0]+1,Final[0]):
                            if Board[i][Initial[1]] != '.':
                                return False
                    else:
                        for i in range(Initial[0]-1,Final[0],-1):
                            if Board[i][Initial[1]] != '.':
                                return False

                    return True
                
            else:
                
                Distance = abs(Final[1] - Initial[1])

                if Distance == 1:
                    return True
                
                else:
                    if Final[1] > Initial[1]:
                        for i in range(Initial[1]+1,Final[1]):
                            if Board[Initial[0]][i] != '.':
                                return False
                    else:
                        for i in range(Initial[1]-1,Final[1],-1):
                            if Board[Initial[0]][i] != '.':
                                return False

                    return True
            
    return False

def Pikachu_moves(Board,Location,player,n):
    moves = list()

    if player == 'w':

        for Loc in Location:
            for k in [1,2]:
                #Down
                if valid_move(Loc[0]+k,Loc[1],n):

                    if Loc[0] + k != n-1:

                        if Board[Loc[0]+k][Loc[1]] == '.' and empty((Loc[0],Loc[1]),(Loc[0]+k,Loc[1]),Board,n,player,'UD'):

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]+k][Loc[1]] = 'W'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)
                        
                        elif Board[Loc[0]+k][Loc[1]] in 'bB' and empty((Loc[0],Loc[1]),(Loc[0]+k,Loc[1]),Board,n,player,'UD'):
                            if valid_move(Loc[0]+k+1,Loc[1],n) and Board[Loc[0]+k+1][Loc[1]] == '.':

                                if Loc[0] + k + 1 != n-1:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]+k+1][Loc[1]] = 'W'
                                    new_Board[Loc[0]+k][Loc[1]] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                                else:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]+k+1][Loc[1]] = '@'
                                    new_Board[Loc[0]+k][Loc[1]] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                    
                    else:

                        if Board[Loc[0]+k][Loc[1]] == '.' and empty((Loc[0],Loc[1]),(Loc[0]+k,Loc[1]),Board,n,player,"UD"):

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]+k][Loc[1]] = '@'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)
                        
                #Right and Left
            for k in [-2,-1,1,2]:
                if valid_move(Loc[0],Loc[1] + k,n):
                        
                        if Board[Loc[0]][Loc[1]+ k] == '.' and empty(Loc,(Loc[0],Loc[1]+ k),Board,n,player,"LR"):
                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]][Loc[1]+k] = 'W'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)

                        elif Board[Loc[0]][Loc[1]+k] in 'bB' and empty(Loc,(Loc[0],Loc[1]+k),Board,n,player,"LR"):
                            x = -1 if k < 0 else 1
                            if valid_move(Loc[0],Loc[1]+k+x,n) and Board[Loc[0]][Loc[1]+k+x] == '.':
                                new_Board = copy.deepcopy(Board)
                                new_Board[Loc[0]][Loc[1]+k+x] = 'W'
                                new_Board[Loc[0]][Loc[1]+k] = '.'
                                new_Board[Loc[0]][Loc[1]] = '.'
                                moves.append(new_Board)

    else:

        for Loc in Location:
            for k in [1,2]:
                #Up
                if valid_move(Loc[0]-k,Loc[1],n):

                    if Loc[0] - k != 0:

                        if Board[Loc[0]-k][Loc[1]] == '.' and empty((Loc[0],Loc[1]),(Loc[0]-k,Loc[1]),Board,n,player,'UD'):

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]-k][Loc[1]] = 'B'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)
                        
                        elif Board[Loc[0]-k][Loc[1]] in 'wW' and empty((Loc[0],Loc[1]),(Loc[0]-k,Loc[1]),Board,n,player,'UD'):
                            if valid_move(Loc[0]-k-1,Loc[1],n) and Board[Loc[0]-k-1][Loc[1]] == '.' :

                                if Loc[0] - k - 1 != 0:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]-k-1][Loc[1]] = 'B'
                                    new_Board[Loc[0]-k][Loc[1]] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                                else:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]-k-1][Loc[1]] = '$'
                                    new_Board[Loc[0]-k][Loc[1]] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                    
                    else:

                        if Board[Loc[0]-k][Loc[1]] == '.' and empty((Loc[0],Loc[1]),(Loc[0]-k,Loc[1]),Board,n,player,'UD'):

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]-k][Loc[1]] = '$'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)
                                
            for k in [-2,-1,1,2]:
                if valid_move(Loc[0],Loc[1] + k,n):
                        
                        if Board[Loc[0]][Loc[1]+ k] == '.' and empty(Loc,(Loc[0],Loc[1]+ k),Board,n,player,"LR"):
                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]][Loc[1]+k] = 'B'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)

                        elif Board[Loc[0]][Loc[1]+k] in 'wW' and empty(Loc,(Loc[0],Loc[1]+k),Board,n,player,"LR"):
                            x = -1 if k < 0 else 1
                            if valid_move(Loc[0],Loc[1]+k+x,n) and Board[Loc[0]][Loc[1]+k+x] == '.':
                                new_Board = copy.deepcopy(Board)
                                new_Board[Loc[0]][Loc[1]+k+x] = 'B'
                                new_Board[Loc[0]][Loc[1]+k] = '.'
                                new_Board[Loc[0]][Loc[1]] = '.'
                                moves.append(new_Board)
    
    return moves

def Pichu_moves(Board,Location, player,n):
    moves = list()

    if player == 'w':

        for Loc in Location:
            for k in [1,-1]:
                if valid_move(Loc[0]+1,Loc[1]+k,n):

                    if Loc[0] + 1 != n-1:

                        if Board[Loc[0]+1][Loc[1]+k] == '.':

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]+1][Loc[1]+k] = 'w'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)

                        elif Board[Loc[0]+1][Loc[1]+k] == 'b':

                            if valid_move(Loc[0]+2,Loc[1]+2*k,n) and Board[Loc[0]+2][Loc[1]+2*k] == 'b':

                                if Loc[0] + 2 != n-1:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]+2][Loc[1]+2*k] = 'w'
                                    new_Board[Loc[0]+1][Loc[1]+k] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                                
                                else:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]+2][Loc[1]+2*k] = '@'
                                    new_Board[Loc[0]+1][Loc[1]+k] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)

                    else:

                        if Board[Loc[0]+1][Loc[1]+k] == '.':

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]+1][Loc[1]+k] = '@'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)
    
    else:
        for Loc in Location:
            for k in [1,-1]:
                if valid_move(Loc[0]-1,Loc[1]+k,n):

                    if Loc[0] - 1 != 0:

                        if Board[Loc[0]-1][Loc[1]+k] == '.':

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]-1][Loc[1]+k] = 'b'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)

                        elif Board[Loc[0]-1][Loc[1]+k] == 'w':

                            if valid_move(Loc[0]-2,Loc[1]+2*k,n) and Board[Loc[0]-2][Loc[1]+2*k] == '.':

                                if Loc[0] - 2 != 0:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]-2][Loc[1]+2*k] = 'b'
                                    new_Board[Loc[0]-1][Loc[1]+k] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)

                                else:
                                    new_Board = copy.deepcopy(Board)
                                    new_Board[Loc[0]-2][Loc[1]+2*k] = '$'
                                    new_Board[Loc[0]-1][Loc[1]+k] = '.'
                                    new_Board[Loc[0]][Loc[1]] = '.'
                                    moves.append(new_Board)
                
                    else:

                        if Board[Loc[0]-1][Loc[1]+k] == '.':

                            new_Board = copy.deepcopy(Board)
                            new_Board[Loc[0]-1][Loc[1]+k] = '$'
                            new_Board[Loc[0]][Loc[1]] = '.'
                            moves.append(new_Board)                               

    return moves

def generate_moves(Board2D,player):

    moves = list()
    Loc,n = Locations(Board2D,player)

    if Loc[2]:
        moves.extend(Raichu_moves(Board2D,Loc[2], player, n))
    
    if Loc[1]:
        moves.extend(Pikachu_moves(Board2D,Loc[1], player, n))

    if Loc[0]:
        moves.extend(Pichu_moves(Board2D,Loc[0], player, n))
    
    return moves

def find_best_move(board, N, player, timelimit):
    #board ='.....$.......WW....w...w..w.......$b..........b.B..B...B...@....'
    depth = 1
    #board = np.array(tuple(board)).reshape(N, N).tolist() #change here

    
    def max_value(board, N, player, depth, alpha, beta,Initial):
        if player == 'w':
            next_player = 'b'
        else:
            next_player = 'w'

        board = np.array(tuple(board)).reshape(N, N).tolist()

        if depth == 0 or check_mate(board,next_player):
            return evaluate_function(board, Initial,depth), None
        

        
        best_score = -float('inf')
        best_move = None
        for move in generate_moves(board, player):

            new_score,_  = min_value(move, N, next_player, depth - 1, alpha, beta,Initial)

            if new_score > best_score:
                best_score = new_score
                best_move =  move
                alpha = max(alpha, best_score)

            if best_score >= beta:
                return best_score, best_move
            
        if best_move is None:
            return evaluate_function(board, Initial,depth), None
        
        return best_score, best_move

    def min_value(board, N, player, depth, alpha, beta,Initial):

        if player == 'w':
            next_player = 'b'
        else:
            next_player = 'w'

        board = np.array(tuple(board)).reshape(N, N).tolist()

        if depth == 0 or check_mate(board,next_player):
            return evaluate_function(board, Initial,depth), None

    
        best_score = float('inf')
        best_move = None
        for move in generate_moves(board, player):

            new_score,_  = max_value(move, N, next_player, depth - 1, alpha, beta,Initial)

            if new_score < best_score:
                best_score = new_score
                best_move = move
                beta = min(beta, best_score)

            if best_score <= alpha:
                return best_score, best_move
        
        if best_move is None:
            return evaluate_function(board, Initial,depth), None
            
        return best_score, best_move
    
    while depth < 10:
        _, move = max_value(board, N, player, depth,-float('inf'), float('inf'),player)
        depth = depth +1
        yield  Print_board(move)
    
    '''_, move = max_value(board, N, player, 6,-float('inf'), float('inf'),player)
    return move'''

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board,N, player, timelimit):
        print(new_board)
