#!/usr/local/bin/python3
# route.py : Find routes through maps
#
# Code by: Manikanta Kodandapani Naidu (k11)
#
# Based on skeleton code by B551 Course Staff, Fall 2023
#


# !/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from math import tanh
import heapq
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics.pairwise import haversine_distances

def find_nearby_city(city_name, road_dataset, cost_function):
    next_cities = []
    next_cities += road_dataset.loc[(road_dataset['from'] == city_name), ['to','miles','speed','segment']].values.tolist()
    next_cities += road_dataset.loc[(road_dataset['to'] == city_name), ['from','miles','speed','segment']].values.tolist()

    next_cities_dict = {}
    for city in next_cities:
        next_cities_dict[city[0]] = city[1]

    # for city in next_cities:
    #     if cost_function == 'segments':
    #         next_cities_dict[city[0]] = city[1]
    #     elif cost_function == 'distance':
    #         next_cities_dict[city[0]] = city[1]
    #     elif cost_function == 'time':
    #         next_cities_dict[city[0]] = city[1]/city[2]
    #     else:
    #         next_cities_dict[city[0]] = city[1]/city[2]

    return next_cities_dict


def find_heuristic(point1, point2):
    
    point1_radians = [radians(point1[0]), radians(point1[1])]
    point2_radians = [radians(point2[0]), radians(point2[1])]

    hav_val = haversine_distances([point2_radians, point1_radians])
    distance = hav_val * 3958.8

    distance = (np.sqrt(np.abs(np.linalg.det(np.array(distance)))))

    return distance
        

def get_route(start, end, cost):
    
    """
    Find shortest driving route between start city and end city
    based on a cost function.

    1. Your function should return a dictionary having the following keys:
        -"route-taken" : a list of pairs of the form (next-stop, segment-info), where
           next-stop is a string giving the next stop in the route, and segment-info is a free-form
           string containing information about the segment that will be displayed to the user.
           (segment-info is not inspected by the automatic testing program).
        -"total-segments": an integer indicating number of segments in the route-taken
        -"total-miles": a float indicating total number of miles in the route-taken
        -"total-hours": a float indicating total amount of time in the route-taken
        -"total-delivery-hours": a float indicating the expected (average) time 
                                   it will take a delivery driver who may need to return to get a new package
    2. Do not add any extra parameters to the get_route() function, or it will break our grading and testing code.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """
    #Loading road segments and GPS data from the txt files
    road_data = pd.read_csv('road-segments.txt', sep=' ', header=None, names=['from','to','miles','speed','segment'])
    geo_data = pd.read_csv('city-gps.txt', sep=' ', header=None, names=['city','latitude','longitude'])

    from_city_gps = geo_data[geo_data['city'] == str(start)].values.tolist()
    end_city_gps = geo_data[geo_data['city'] == str(end)].values.tolist()
    

    fringe=[(0, start, [0,0,0,0,[], from_city_gps])]

    visited = {start:find_heuristic(from_city_gps[0][1:], end_city_gps[0][1:])}

    while fringe:
        (h_val, from_city, properties) = heapq.heappop(fringe)

        next_cities = []
        next_cities += road_data.loc[(road_data['from'] == from_city), ['to','miles','speed','segment']].values.tolist()
        next_cities += road_data.loc[(road_data['to'] == from_city), ['from','miles','speed','segment']].values.tolist()

        for city in next_cities:
            new_to_city_gps = geo_data[geo_data['city'] == str(city[0])].values.tolist()

            new_segment_count = properties[0] + 1
            new_miles_count = properties[1] + city[1]
            new_time = properties[2] + (city[1]/city[2])
            new_route = properties[4] + [("{}".format(city[0]),"{} for {} miles".format(city[3], city[1])),]
            if city[2] >= 50:
                new_delivery_time = properties[3] + (tanh(city[1]/1000))*2*(new_time) + (city[1]/city[2])
            else:
                new_delivery_time = properties[3] + (city[1]/city[2])

            if city[0] == end:
                return {"total-segments" : new_segment_count, 
                        "total-miles" : float(new_miles_count), 
                        "total-hours" : new_time, 
                        "total-delivery-hours" : new_delivery_time, 
                        "route-taken" : new_route}

            city_found = True

            if not new_to_city_gps:
                nearby_cities = find_nearby_city(city[0], road_data, cost)
                nearest_city = min(nearby_cities, key=nearby_cities.get)

                while city_found:
                    new_to_city_gps = geo_data[geo_data['city'] == str(nearest_city)].values.tolist()

                    if len(new_to_city_gps)>0:
                        city_found = False
                    else:
                        if len(nearby_cities)>1:
                            del nearby_cities[nearest_city]
                            nearest_city = min(nearby_cities, key=nearby_cities.get)
                        else:
                            new_to_city_gps = properties[5]
                            city_found = False

            if city_found:
                heuristic_val = find_heuristic(new_to_city_gps[0][1:], end_city_gps[0][1:])
            else:
                if len(nearby_cities)>0:
                    heuristic_val = find_heuristic(new_to_city_gps[0][1:], end_city_gps[0][1:]) + nearby_cities[nearest_city]
                else:
                    heuristic_val = find_heuristic(new_to_city_gps[0][1:], end_city_gps[0][1:])

            if cost == "segments":
                new_h_val = new_segment_count
            elif cost == "distance":
                new_h_val = new_miles_count + heuristic_val
            elif cost == "time":
                new_h_val = new_time + (heuristic_val/city[2])
            elif cost == "delivery":
                if city[2]>=50:
                    new_h_val =  new_delivery_time + (heuristic_val/50)
                else:
                    new_h_val =  new_delivery_time + (heuristic_val/city[2])
            else:
                pass

            if city[0] not in visited or new_h_val < visited[city[0]]:
                    heapq.heappush(fringe, (new_h_val, city[0], [new_segment_count, new_miles_count, new_time, new_delivery_time, new_route, new_to_city_gps]))
                    visited[city[0]] = new_h_val

    return "No route found"

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise(Exception("Error: expected 3 arguments"))

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "delivery"):
        raise(Exception("Error: invalid cost function"))

    result = get_route(start_city, end_city, cost_function)

    # Pretty print the route
    print("Start in %s" % start_city)
    for step in result["route-taken"]:
        print("   Then go to %s via %s" % step)

    print("\n          Total segments: %4d" % result["total-segments"])
    print("             Total miles: %8.3f" % result["total-miles"])
    print("             Total hours: %8.3f" % result["total-hours"])
    print("Total hours for delivery: %8.3f" % result["total-delivery-hours"])


