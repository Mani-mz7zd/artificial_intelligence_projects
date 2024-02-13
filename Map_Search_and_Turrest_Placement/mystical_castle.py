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
        current_loc=[(row_i,col_i) for col_i in range(len(castle_map[0])) for row_i in range(len(castle_map)) if castle_map[row_i][col_i]=="p"][0]
        #initializing direction value to '' for the starting point.
        if castle_map[current_loc[0]][current_loc[1]]=='@':
               return (0,"")
        fringe=[(current_loc,0,'')]

        #array for tracking the visited location so that the algorithms does not end up in infinite while loop 
        visited_loc = []
        #array for tracking the path directions
        loc_moves = []

        while fringe:
                #In order to find the shortest path, we are using BFS approach where Queue data structure is used.
                #Therefore, first element from the fringe list has to be poped for searching
                (curr_move, curr_dist, curr_path)=fringe.pop(0)
                for move in moves(castle_map, curr_move[0],curr_move[1]):
                        if (move[0],move[1]) not in visited_loc: #checking if the location is already visited. If not visited, proceed 
                            if castle_map[move[0]][move[1]]=="@":
                                return (curr_dist+1, curr_path+move[2]) #returns the travel distance and path
                            else:
                                fringe.append((move, curr_dist + 1,curr_path+move[2])) #appends the next set of locations to visit
                                visited_loc.append((move[0],move[1])) #appends the visited location to the array for tracking
        return (-1,"")

# Main Function
if __name__ == "__main__":
        castle_map=parse_map(sys.argv[1])
        print("Shhhh... quiet while I navigate!")
        solution = search(castle_map)
        print("Here's the solution I found:")
        print(str(solution[0]) + " " + solution[1])

