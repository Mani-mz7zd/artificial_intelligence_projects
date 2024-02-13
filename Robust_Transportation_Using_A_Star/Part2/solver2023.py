#!/usr/local/bin/python3
# solver2023.py : 2023 Sliding tile puzzle solver
#
# Code by: Bhagath Singh Bondili (bbondili)
#
# Based on skeleton code by B551 Staff, Fall 2023

import sys
import test_a1_2023_puzzle
import copy
import numpy as np
import heapq

ROWS = 5
COLS = 5

def inner_board_c(board):
    board=np.array(board)
    inner_board=board[1:-1,1:-1].tolist()
    inner_board = test_a1_2023_puzzle.move_clockwise(inner_board)
    board[1:-1,1:-1]=np.array(inner_board)
    board=board.tolist()
    return board


def inner_board_cc(board):
    board=np.array(board)
    inner_board=board[1:-1,1:-1].tolist()
    inner_board = test_a1_2023_puzzle.move_cclockwise(inner_board)
    board[1:-1,1:-1]=np.array(inner_board)
    board=board.tolist()
    return board

def printable_board(board):
    return [(' %2d ')*COLS % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS)]

def successors(state):
    successors_boards = []

    for i in range(ROWS):
        successors_boards.append((test_a1_2023_puzzle.move_left(copy.deepcopy(state), i), "L{}".format(i + 1)))
        successors_boards.append((test_a1_2023_puzzle.move_right(copy.deepcopy(state), i), "R{}".format(i + 1)))

    for i in range(COLS):
        successors_boards.append((test_a1_2023_puzzle.transpose_board(test_a1_2023_puzzle.move_left(test_a1_2023_puzzle.transpose_board(state), i)), "U{}".format(i + 1)))
        successors_boards.append((test_a1_2023_puzzle.transpose_board(test_a1_2023_puzzle.move_right(test_a1_2023_puzzle.transpose_board(state), i)), "D{}".format(i + 1)))

    successors_boards.append((test_a1_2023_puzzle.move_clockwise(copy.deepcopy(state)), "Oc"))
    successors_boards.append((test_a1_2023_puzzle.move_cclockwise(copy.deepcopy(state)), "Occ"))

    successors_boards.append((inner_board_c(copy.deepcopy(state)), "Ic"))
    successors_boards.append((inner_board_cc(copy.deepcopy(state)), "Icc"))

    return successors_boards

def is_goal(state):
    return state == np.array(tuple(list(range(1, 26)))).reshape(ROWS, COLS).tolist()

def manhattan_distance(state):
    distance = 0

    for i in range(ROWS):
        for j in range(COLS):
            if state[i][j] != 0:
                row = (state[i][j] - 1) // ROWS
                col = (state[i][j] - 1) % COLS
                distance += abs(row - i) + abs(col - j)

    return distance * 0.2

def solve(initial_board):
    """
    1. This function should return the solution as instructed in assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing we will call this function with single argument(initial_board) and it should return 
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution
    """
    initial_state_board = np.array(tuple(initial_board)).reshape(ROWS, COLS).tolist()
    fringe = []
    reached = {}

    g_cost = 0  # Actual cost from the start state to the current state
    h_cost = manhattan_distance(initial_state_board)  # Manhattan distance heuristic
    f_cost = g_cost + h_cost

    reached[tuple(map(tuple, initial_state_board))] = f_cost
    heapq.heappush(fringe, (f_cost, [], initial_state_board))

    while len(fringe)>0:
        (f_cost, moves, state) = heapq.heappop(fringe)

        if is_goal(state):
            return moves

        for s in successors(state):
            new_moves = moves + [s[1]]
            new_state = s[0]

            g_cost = len(new_moves)
            h_cost = manhattan_distance(new_state)
            f_cost = g_cost + h_cost

            if tuple(map(tuple, new_state)) not in reached or f_cost < reached[tuple(map(tuple, new_state))]:
                reached[tuple(map(tuple, new_state))] = f_cost
                heapq.heappush(fringe, (f_cost, new_moves, new_state))

    return []

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))
    
    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
