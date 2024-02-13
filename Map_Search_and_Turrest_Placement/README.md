# Assignment 1

## Problem 1 - Searching a map for paths
This problem requires to find the shortest path from point 'p' to point '@' in the provided castle map.
The reasons for the initial code to fail are as follows.
- First of all, this is a BFS (Breath first search) problem because we are required to find the shortest path and thus we explore the entire elements at a particular branch level before traversing to the next level for searching. In BFS, we utilize Queue to implement the search technique where it follows FIFO (first in first out) approach. However in the initial code, the pop() function was used instead of pop(0) because of which the backend searching was DFS instead of BFS.
- Secondly, in this problem, we need to track the locations that have been visited or explored to avoid the functionality ending up in the infinite searching loop. This logic was not incorporated in the initial code which caused the functionality to go into an infinite search loop and had to be stopped through keyboard interrupt.

Resolutions are as follows.
- Implemented Breath First Search (BFS) by modifying the pop() function to pop(0) which resulted in using the queue data structure
 with the FIFO (first in first out) approach. Therefore, all the elements at a particular branch are explored before traversing to the next level of search. This favors in finding the shortest path.
- To keep track of the visited locations/positions, I utilized list data structure.
- Since we need to return the path (directions) as an output, I have modified moves() function to include the direction as well which will be one of ['D', 'U', 'R', 'L'].

Execution Flow is as follows.
- First, we find the location of the point 'p' in the given castle map and search starts from this location which is the first element in the fringe.
- Using BFS search, we find the locations/positions that have either '.' or '@' across all 4 directions (Down, Up, Right and Left) from the fringe location and add it the fringe for further exploration. With this, we are excluding the locations having 'X' because we cannot traverse through 'X' positions.
- We add the explored locations in the list data structure so that we do not end up exploring the same locations again and again.
- Once the destination denoted as '@' is reached, we return the path count and path directions as an output.
- If there is no solution for the given castle map, we return -1 as an output.

The standard formulation of the castle path finder problem is as follows.
- Valid Sates: For this problem of mystical castle, the set of valid states are the states or positions that are either the open space denoted by '.' or the destination denoted by '@'.
- Initial State: The initial state is the starting point within the castle map denoted by 'p'.
- Successor function: The successor function is defined as the set of possible movements depending on the type of the cell from a given position through all four directions such as UP, DOWN, RIGHT and LEFT.
- Goal state: The goal state is to arrive at the destination denoted by '@' traversing through the shortest possible path from its starting point 'p'.
- Cost function: Since this is BFS problem, it is associated with search in each step and the cost associated with it can be considered as 1. This cost is constant untill end goal is achieved.


## Problem 2 - Searching for Design for turrets
This problem requires to find the design for placing the given number of turrets in the castle map. The initial code lacks the logic to check the visibility constraints on the turret placements which resulted in failing to output the results.

Resolution is as follows.
- The add_turret() has been modified to include the logic to check the visibility constraints vertically(row), horizontally(column) and diagonally for the turret placed. As a result of this, only the valid and safe positions satisfying the visibility constraint are return for further exploration. 
- This is a Depth First Search (DFS) problem implemented using Stack data structure with LIFO (Last in first out) approach.

Execution flow is as follows.
- The input castle map is the first element in the fringe and search starts with the initial castle map where exactly one turret is already placed.
- The successor() function returns all the valid position configurations satisfying the visibility constraints. Each of these positions are explored untill we get the required number of turrets placed on the map.
- The add_turret() is the key function where the visibility constraint check logic is incorporated. This logic flow is as follows.
    - row check: Here we check the number of turrets denoted as 'p' in the row. If the count is greater than 1, we check if the wall denoted as 'X' or '@' is present between the turrets.
    - column check: Here we check the number of turrets denoted as 'p' in the column. If the count is greater than 1, we check if the wall denoted as 'X' or '@ is present between the turrets.
    - diagonal check: Here we check the number of turrets denoted as 'p' in both diagonal and anti-diagonal directions separately. If the count is greater than 1, we check if the wall denoted as 'X' or '@' is present between the turrets.
- If all the above checks are safe, we return the respective position configuration in the returned list, otherwise we exclude it from the list. With this logic, the successor() functions returns only the valid and safe positions list for further exploration. Please note that 'safe' here means that the turrets won't be able to see each other.
- Once all the given number of turrets are placed, the designed castle map is returned as an output.
- If there is no solution, where we cannot not place the given number of turrets on the provided castle map, we return boolean 'False' as an output.

The standard formulation of the design for turrets problem is as follows
- State space: The state space encompasses all possible positions and configurations for turret placement on the castle map. Each turret placement on the map can be considered a partial solution where some turrets are already placed, and some locations remain empty.
- Initial State: The initial state represents a castle map with exactly one turret placed on it.
- Successor function: The successor function is defined as the set of possible position configuration states where the turrets can be placed and these placements are to be valid, safe and satisfying the mandatory visibility constraints.
- Goal state: The goal state is to achieve the placement of all the given number of turrets and return the designed castle map.
- Cost function: For such a search problem, it would be a good practice to consider the cost associated with each position search for turret placement as 1.
