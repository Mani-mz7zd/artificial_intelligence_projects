# vsshinde-bbondili-k11-a1

# Problem 1:

Description:

The problem involves a group of fairies with magical symbols arranged randomly. The goal is to rearrange them into ascending order from 1 to N using the fewest steps possible. The fairies can swap positions with their neighboring companions in a single step.


### Problem Formulation:
1. State Space: The state space is defined as all possible configurations of the fairies' symbols, represented as a list of integers from 1 to N. Each integer represents a fairy's magical symbol, and the order of the integers in the list represents the current arrangement of fairies.

2. Successor Function: The successor function generates a list of successor states from a given state by swapping the positions of two neighboring fairies. It does this for all possible pairs of neighboring fairies in the current state.

3. Edge Weights: In this problem, edge weights are not explicitly defined since we are not optimizing for a specific cost. However, the number of steps taken to reach the goal state can be used as a measure of solution optimality.

4. Goal State: The goal state is defined as a state in which the fairies are arranged in ascending order from 1 to N. For example, for N=5, the goal state is [1, 2, 3, 4, 5].

5. Heuristic Function: The heuristic function h(state) is designed to estimate the number of steps required to reach the goal state from the given state. It calculates the sum of the Manhattan distances between each fairy's current position and its goal position. The Manhattan distance between two points (x1, y1) and (x2, y2) is given by abs(x1 - x2) + abs(y1 - y2). The heuristic function is admissible because it never overestimates the cost to reach the goal. In this case, the heuristic calculates the distance to move each fairy to its correct position, and the sum of these distances is always less than or equal to the actual number of swaps needed to reach the goal state.

### Search Algorithm:

We choose to use a modified form of Breadth-First Search (BFS) to find the solution. It maintains a priority queue (fringe) where states are sorted based on their estimated cost to the goal using the heuristic function.
The reached dictionary is used to keep track of the states that have been reached along with their estimated costs.
The algorithm explores states in order of increasing estimated cost until it reaches the goal state.
It uses the successor function to generate new states by swapping adjacent fairies and considers the cost of reaching each successor state.
The algorithm terminates when the goal state is found, and it returns the path to the goal along with the cost.


# Problem 2:

Description:

The problem we are dealing with is a variant of the well-known sliding tile puzzle, similar to the 8-puzzle, but with some unique rules and a larger 5x5 grid. You are given a 5x5 grid with 25 tiles numbered from 1 to 25, arranged in a specific initial configuration. The goal of the puzzle is to find a sequence of moves that will restore the grid to its canonical configuration, where the tiles are numbered from 1 to 25 in left-to-right, top-to-bottom order.

Rules:

1. No Empty Spaces: Unlike the traditional sliding tile puzzle where there is an empty space, in this puzzle, there are no empty spots on the board. We can't slide tiles into an open space.
2. Moves: We can make four types of moves:

a. Slide Rows: We can slide an entire row of tiles left or right, with the leftmost or 
rightmost tile "wrapping around" to the other side of the board.

b. Slide Columns: We can slide an entire column of tiles up or down, with the topmost or bottommost tile "wrapping around" to the other side of the board.

c. Rotate Outer Ring: We can rotate the outer "ring" of tiles either clockwise or counterclockwise.

d. Rotate Inner Ring: We can rotate the inner "ring" of tiles either clockwise or counterclockwise.

### Problem formulation:

1. State Space: The state space consists of all possible arrangements of the 5x5 grid in the puzzle. Each state represents a unique configuration of the board, with each cell containing a tile numbered from 1 to 25. These states collectively define the entire range of possible board configurations for the puzzle.

2. Successor Function: The successor function generates possible successor states from the current state by applying valid moves:
* Sliding entire rows left or right (wrapping around if necessary).
* Sliding entire columns up or down (wrapping around if necessary).
* Rotating the outer ring of tiles (clockwise or counterclockwise).
* Rotating the inner ring of tiles (clockwise or counterclockwise).

The successors(state) function returns a list of successor states and the corresponding actions.

3. Edge Weights: In this puzzle, all edge weights are uniform. Each move from one state to another has a cost of 1.

4. Goal State: The goal state is defined as the canonical configuration where the tiles are numbered from 1 to 25 in left-to-right, top-to-bottom order.

5. Heuristic Function:  The heuristic function, heuristic(state), estimates the number of steps needed to reach the goal state from the current state.

* Initialization: For each tile in the current state, the heuristic function calculates the Manhattan distance to its goal position. It is the sum of the absolute horizontal and vertical differences between the current position and the goal position of the tile.

* Vertical and horizontal distances: The function considers both the vertical and horizontal distance separately for each tile. It actually is calulating how many rows ad columnseach tile is away from its actual goal position.

* Warpping around conditions: In this condition, in the puzzle, when sliding a row or a column, the leftmost or the topmost tile wraps around the rightmost or the bottommost position, respectivley. So to handle this, the function is calculating the minimum distance by considering both the direct distance and the wrapped distance.

The function then is computing the sum of the vertical and horizontal distances for all the tiles. It represents the total Manhattan distance between the current state and the goal state.

**Search Algorithm**

The search algorithm implemented for this problem statement is an informed search strategy known as A* search algorithm.It efficiently explores the state space to find a solution to the 2023 puzzle by systematically evaluating and prioritizing states based on their estimated costs.

* The algoritm starts by initializing essential data structures, including priority queue ('fringe') to manage the states and also a dictionary: 'reached', to track cost estimates.
* While 'finge' is not empty: 

The state with the lowest estimates total cost ('g_cost +h_val') is removed from the fringe.
If the state corresponds to the goal state, the algorithm terminates and returns the sequence of the moves leading to the goal.

Otherwise, the algorithm geneates possible successor states from the current state using the 'successors(state)' function. It calculates the cost to reach each successor state and updates their cost estimates. Successor states that haven't been explored before or have a lower cost estimate are added to the fringe for further exploration.

Q 1. In this problem, what is the branching factor of the search tree?

Ans: the branching factor of the search tree can be determined by considering the number of possible successor states for each state. Let's break it down:

a. Sliding rows left and right: For each of the 5 rows, you can slide it left or right, resulting in 2 actions per row. So, there are a total of 5 rows * 2 actions/row = 10 possible states from row movements.

b. Sliding columns up and down: Similarly, for each of the 5 columns, you can slide it up or down, resulting in 2 actions per column. So, there are a total of 5 columns * 2 actions/column = 10 possible states from column movements.

c. Rotating the outer ring: There are two actions for this, clockwise and counterclockwise.

d. Rotating the inner ring: Again, two actions for this, clockwise and counterclockwise.

So, the total branching factor for this problem would be 10 (row movements) + 10 (column movements) + 2 (outer ring rotations) + 2 (inner ring rotations) = 24.

Q.2 If the solution can be reached in 7 moves, about how many states would we need to explore before we found it if we used BFS instead of A* search? A rough answer is fine.

Ans: If the solution can be reached in 7 moves, and we use Breadth-First Search (BFS), we would need to explore a significant number of states before finding the solution. In BFS, all nodes at a given depth are explored before moving to the next depth. Since the branching factor is 24, for each level of depth, we would explore 24 times more states than the previous level.

For a depth of 1 (starting state), we would explore 24 state.
For a depth of 2, we would explore 24^2 states.
For a depth of 3, we would explore 24^3 states.
...
For a depth of 7 (the solution), we would explore 24^7 states.

Calculating this, we get:
1+24+(24^2)+..+(24^7) = O(24^7) (i.e around 4586471424 states).


# Problem 3:

Description: 

The given probem statement asks us to find the optimal driving route between two cities un the United States, considering various cost functions.The goal is to efficiently transport valuable artifacts and treasures from one city to another while taking into account different cost criteria, such as the number of road segments, total distance, total time, and a specialized delivery driver scenario.

### Problem formulation:

State space: 

* Cities and Towns: Each city or town in the dataset represents a node in the state space.

* Road Segments: The road segments connecting cities represent the edges in the state space. Each road segment is associated with various attributes, such as distance, speed limit, and segment description.

* Current City: The algorithm is in one of the cities at any given moment, and this city represents the current state within the state space.

Successor Function:

The successor function generates possible next cities to explore based on the current city. It considers road segments starting from or leading to the current city. The successor function returns the neighboring cities, their distances, speeds, and segment descriptions. Depending on the cost function, these values are used to calculate the cost of moving from one city to another.

Goal State: 

The goal state is reached when the current city is the same as the specified end city. At this point, the algorithm returns the optimal route and related information.

Heuristic Function Description:

The heuristic function used in this code is based on the Haversine distance formula. The Haversine formula is commonly used in geospatial applications to calculate the great-circle distance between two points on the Earth's surface. It takes into account the latitude and longitude coordinates of two locations and provides an estimate of the shortest distance between them, considering the curvature of the Earth. The haversine_distances function from the sklearn.metrics.pairwise module calculates the haversine distance between point1 and point2 when provided the latitude and longitudes of the 2 points.The haversine distance is initially calculated in kilometers and to convert it to miles, we multiply it by the Earth's radius in miles, which is approximately 3958.8 miles. This is a common conversion factor used in haversine calculations. The  square root of the absolute value ensures that the heuristic always returns a positive value.

The heuristic guides the A* search algorithm in prioritizing which cities to explore next. When choosing the next city to visit, the algorithm considers both the known cost from the start city to the current city and the estimated remaining cost from the current city to the destination city (heuristic value). the heuristic provides a lower bound on the remaining distance. This property is essential for the A* search algorithm's correctness. Since the Earth's surface distance is a valid lower bound on travel distance, the Haversine heuristic is admissible.

Problems, Assumptions, and Design Decisions:
Cases where there's no GPS data for a city are handled by finding nearby cities with road data. This is a reasonable approach, but it might lead to suboptimal routes in some cases.

