import numpy as np
import pandas as pd


class Goose:

    all_geese = {}
    active_geese = {}

    @staticmethod
    def distance_between_geese(goose1, goose2):
        """Calculate the distance between two geese"""
        distance = np.linalg.norm(goose1.position - goose2.position)
        return distance

    def __init__(self, trj_id, xpos, ypos, zpos, ax):
        """Initialize a goose instance with an id and position"""
        self.trj_id = trj_id
        self.position = np.array([xpos, ypos, zpos])
        # self.update_flock()
        Goose.all_geese[trj_id] = self
        Goose.active_geese[trj_id] = self
        (self.location_plotter,) = ax.plot(
            [xpos],
            [ypos],
            [zpos],
            "o",
            color="red",
        )  # TODO: Make this color the flock color # matplotlib object
        (self.history_plotter,) = ax.plot3D([], [], [], color="red", linewidth=".4")
        self.historical_flight_path = {
            "xlocs": [xpos],
            "ylocs": [ypos],
            "zlocs": [zpos],
        }  # dict of structure {xlocs: [], ylocs: [], zlocs: []}
        self.absent_counter = 0  # if this counter hits 60, the goose has not been detected moving for one second

    def find_closest_neighbor(self):
        """finds the closest neighbor and returns it and the distance"""
        closest_neighbor = None
        distance_to_closest_neighbor = np.inf

        # check distances to all other geese
        for goose in Goose.active_geese:
            distance = distance_between_geese(self, goose)
            if distance < distance_to_closest_neighbor:
                # update closest neighbor
                closest_neighbor = goose
                distance_to_closest_neighbor = distance

        return closest_neighbor, distance

    def update_position(self, xpos, ypos, zpos):
        """update goose postion and flock it belongs to"""

        # update its position
        self.position = np.array([xpos, ypos, zpos])

        # save position in historical flight data
        self.historical_flight_path["xlocs"].append(xpos)
        self.historical_flight_path["ylocs"].append(ypos)
        self.historical_flight_path["zlocs"].append(zpos)

        # update flock it belongs to
        # self.update_flock()

        return True

    def update_flock(self):
        """Update the flock that the goose is a part of and the flock's parameters"""
        # update parameters of goose
        self.flock = self.determine_flock()
        # update parameters of flock
        self.flock.add_goose(self)
        return True

    def determine_flock(self):
        """determine the flock a goose belongs to"""
        closest_goose = None
        distance_to_closest_goose = np.inf

        for flock in Flock.list_of_flocks:
            for goose in flock.geese:
                # exclude itself as closest neighbor
                if goose is not self:
                    distance = Goose.distance_between_geese(self, goose)
                    if distance < distance_to_closest_goose:
                        closest_goose = goose
                        distance_to_closest_goose = distance

        # if no goose exists yet
        if closest_goose == None:
            # create a new flock
            return Flock()

        # if the closest goose has not been assigned a flock yet, make an own flock
        if closest_goose.flock == None:
            return Flock()

        # logic of creating new flocks if a goose is too far away from other flocks
        if len(closest_goose.flock.geese) < 2:
            if distance_to_closest_goose > 1000:
                # create a new flock
                return Flock()
        else:
            if distance_to_closest_goose > closest_goose.flock.distance_mean * 3:
                # create a new flock
                return Flock()

        return closest_goose.flock

    def deactivate(self):
        """Deactivate a Goose because it is not moving any more"""
        del Goose.active_geese[self.trj_id]
        # self.flock.geese.remove(self)
        # self.flock = None
        self.location_plotter.remove()
        return True


class Flock:

    list_of_possible_colors = ["blue", "green", "brown", "purple"]
    color_counter = 0
    list_of_flocks = []

    def __init__(self):

        # TODO: mean and std dev = 0 might be a problem!
        self.geese = []  # list of geese that are part of this flock
        self.distance_mean = 0  # mean distance of geese to the next goose
        self.distance_std_dev = 0  # std dev of geese distances to the next goose
        Flock.list_of_flocks.append(self)
        self.color = Flock.list_of_possible_colors[Flock.color_counter]
        Flock.color_counter = (Flock.color_counter + 1) % len(
            Flock.list_of_possible_colors
        )

    def add_goose(self, goose):
        self.geese.append(goose)
        self.update_distance_mean()
        return True

    def remove_goose(self, goose):
        self.geese.remove(goose)
        self.update_distance_mean()
        return True

    def update_distance_mean(self):
        distance_sum = 0
        for goose in self.geese:
            minimum_distance = np.inf
            for neighbor in self.geese:
                if goose is not neighbor:
                    distance = Goose.distance_between_geese(goose, neighbor)
                    if distance < minimum_distance:
                        minimum_distance = distance

            # add minimum distance to total distance sum
            distance_sum += minimum_distance

        # calculate average
        self.distance_mean = distance_sum / len(self.geese)

        return True
