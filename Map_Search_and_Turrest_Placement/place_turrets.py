#!/usr/local/bin/python3
#
# place_turrets.py : arrange turrets on a grid, avoiding conflicts
#
# Submitted by : Manikanta Kodandapani Naidu, k11
#
# Based on skeleton code in CSCI B551, Fall 2022.

import sys

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]

# Count total # of turrets on castle_map
def count_turrets(castle_map):
    return sum([ row.count('p') for row in castle_map ] )

# Return a string with the castle_map rendered in a human-turretly format
def printable_castle_map(castle_map):
    return "\n".join(["".join(row) for row in castle_map])

# Add a turret to the castle_map at the given position, and return a new castle_map (doesn't change original)
def add_turret(castle_map, row, col):

    res = castle_map[0:row] + [castle_map[row][0:col] + ['p',] + castle_map[row][col+1:]] + castle_map[row+1:]

    #finding the vertical, horizontal and diagonal paths to check for safety
    check_list =[]
    check_list.append(res[row])
    check_list.append([res[row_value][col] for row_value in range(len(res))])

    if row > col:
        d1_row_index_list = [i for i in range((row-col),len(res))]
        d1_col_index_value = [j for j in range(len(d1_row_index_list))]
        check_list.append([res[d1_row_index_list[a]][d1_col_index_value[a]] for a in range(len(d1_row_index_list))])

    elif col > row:
        col_index_value = [i for i in range((col-row),len(res[0]))]
        row_index_list = [j for j in range(len(col_index_value))]
        check_list.append([res[row_index_list[a]][col_index_value[a]] for a in range(len(row_index_list))])

    else:
        max_index = min(len(res),len(res[0]))
        check_list.append([res[i][i] for i in range(max_index)])

    max_row_index = len(castle_map)-1
    max_col_index = len(castle_map[0])-1

    start_row_1 = row+1
    start_col_1 = col-1
    start_row_2 = row-1
    start_col_2 = col+1

    D2 = []
    while start_row_1<=max_row_index and start_col_1>=0:
        D2.append(res[start_row_1][start_col_1])
        start_row_1 += 1
        start_col_1 -= 1
    D2.append(res[row][col])


    while start_row_2>=0 and start_col_2<=max_col_index:
        D2.append(res[start_row_2][start_col_2])
        start_row_2 -= 1
        start_col_2 += 1

    check_list.append(D2)

    #checking if the vertical, horizontal and diagonal paths are safe
    for element in check_list:
        if element.count('p') > 1:
            p_index_list = []
            for i in range(len(element)):
                if element[i] == 'p':
                     p_index_list.append(i)
            
            for i in range(len(p_index_list)-1):
                if ('X' not in element[p_index_list[i]+1:p_index_list[i+1]]) and ('@' not in element[p_index_list[i]+1:p_index_list[i+1]]):
                    return []
    return res

# Get list of successors of given castle_map state
def successors(castle_map):
    return [ add_turret(castle_map, r, c) for r in range(0, len(castle_map)) for c in range(0,len(castle_map[0])) if castle_map[r][c] == '.' ]

# check if castle_map is a goal state
def is_goal(castle_map, k):
    return count_turrets(castle_map) == k 

# Arrange turrets on the map
#
# This function MUST take two parameters as input -- the castle map and the value k --
# and return a tuple of the form (new_castle_map, success), where:
# - new_castle_map is a new version of the map with k turrets,
# - success is True if a solution was found, and False otherwise.
#
def solve(initial_castle_map,k):
    fringe = [initial_castle_map]
    while len(fringe) > 0:
        for new_castle_map in successors( fringe.pop() ):
            if is_goal(new_castle_map,k):
                return(new_castle_map,True)
            fringe.append(new_castle_map)

    return (initial_castle_map, False)

# Main Function
if __name__ == "__main__":
    castle_map=parse_map(sys.argv[1])
    # This is k, the number of turrets
    k = int(sys.argv[2])
    print ("Starting from initial castle map:\n" + printable_castle_map(castle_map) + "\n\nLooking for solution...\n")
    solution = solve(castle_map,k)
    print ("Here's what we found:")
    print (printable_castle_map(solution[0]) if solution[1] else "False")
