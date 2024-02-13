#!/usr/local/bin/python3
#
# mystical_castle.py : a maze solver
#
# Submitted by : Manikanta Kodandapani Naidu, k11
#
# Based on skeleton code provided in CSCI B551, Fall 2023.

import sys

# Parse the map from a given filename
def parse_map(filename):
        with open(filename, "r") as f:
                return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]
                
# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
        #added 3rd parameter to fetch the direction details which will be one out of ['U','D','R','L']
        moves=((row+1,col,'D'), (row-1,col,'U'), (row,col-1,'L'), (row,col+1,'R'))

        # Return only moves that are within the castle_map and legal (i.e. go through open space ".")
        return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" )]

# Perform search on the map
#
# This function MUST take a single parameter as input -- the castle map --
# and return a tuple of the form (move_count, move_string), where:
# - move_count is the number of moves required to navigate from start to finish, or -1
#    if no such route exists
# - move_string is a string indicating the path, consisting of U, L, R, and D characters
#    (for up, left, right, and down)

def search(castle_map):
        # Find current start position
        current_loc_start=[(row_i,col_i) for col_i in range(len(castle_map[0])) for row_i in range(len(castle_map)) if castle_map[row_i][col_i]=="p"][0]
        current_loc_dest=[(row_i,col_i) for col_i in range(len(castle_map[0])) for row_i in range(len(castle_map)) if castle_map[row_i][col_i]=="@"][0]
        #initializing direction value to '' for the starting point.
        fringe_start=[(current_loc_start,0,'')]
        fringe_end=[(current_loc_dest,0,'')]

        #array for tracking the visited location so that the algorithms does not end up in infinite while loop 
        visited_loc_start = {}
        visited_loc_end = {}
        #array for tracking the path directions

        while fringe_start or fringe_end:
                #In order to find the shortest path, we are using BFS approach where Queue data structure is used.
                #Therefore, first element from the fringe list has to be poped for searching
                (curr_move, curr_dist, curr_path)=fringe_start.pop(0)
                for move in moves(castle_map, curr_move[0],curr_move[1]):
                        if (move[0],move[1]) not in visited_loc_start.keys(): #checking if the location is already visited. If not visited, proceed 
                            if (move[0],move[1]) in visited_loc_end.keys():
                                return (curr_dist + visited_loc_end[(move[0],move[1])][0], curr_path+move[2]+visited_loc_end[(move[0],move[1])][1]) #returns the travel distance and path
                            else:
                                fringe_start.append((move, curr_dist + 1,curr_path+move[2])) #appends the next set of locations to visit
                                visited_loc_start[((move[0],move[1]))] = [curr_dist+1, curr_path+move[2]] #appends the visited location to the array for tracking

                (curr_move_1, curr_dist_1, curr_path_1)=fringe_end.pop(0)
                for move in moves(castle_map, curr_move_1[0],curr_move_1[1]):
                        if (move[0],move[1]) not in visited_loc_end.keys(): #checking if the location is already visited. If not visited, proceed 
                            if (move[0],move[1]) in visited_loc_start.keys():
                                return (curr_dist_1 + visited_loc_start[(move[0],move[1])][0], curr_path_1+move[2]+visited_loc_start[(move[0],move[1])][1]) #returns the travel distance and path
                            else:
                                fringe_end.append((move, curr_dist + 1,curr_path+move[2])) #appends the next set of locations to visit
                                visited_loc_end[((move[0],move[1]))] = [curr_dist_1+1, curr_path_1+move[2]] #appends the visited location to the array for tracking

        return (-1,"")

# Main Function
if __name__ == "__main__":
        castle_map=parse_map(sys.argv[1])
        print("Shhhh... quiet while I navigate!")
        solution = search(castle_map)
        print("Here's the solution I found:")
        print(str(solution[0]) + " " + solution[1])

