"""obstacle_tracker.py
"""

import cv2
import numpy as np
from collections import OrderedDict
import logging
from abc import ABC, abstractmethod
from .semantic_mapper import ObjectType
from .utils import *
from .binarize import *

logger = logging.getLogger(__name__)

class ObstacleTracker:
    """ ObstacleTracker filter obstacles (namely ducks, cones and robots) detected by the SemanticMapper by associating each bbox throughout multiple frames.
    """
    def __init__(self, min_frames=5, max_frames=1, gating_t=200, lbf_t=100):
        self.min_frames = min_frames # Minimum no. frames in which the object must be present in order to be considered a concrete obstacle
        self.max_frames = max_frames # Maximum no. frames in which the object can disappear before it is considered disappeared
        self.gating_t = gating_t # Gating threshold
        self.lbf_t = lbf_t # Lonely Best Friend threshold
        self.curr_obstacles = None # Current frame's obstacles
        self.prev_obstacles = None # Previous frame's obstacles
        self.obstacles = OrderedDict()
        self.appeared = OrderedDict()
        self.disappeared = OrderedDict()
        self.oid = 0

    def _extractObstacles(self, semantic_map):
        """ Extract dangerous objects from the pre-computed semantic map
        """
        obstacle_lst = []
        obstacle_keys = [ObjectType.DUCK, ObjectType.CONE, ObjectType.ROBOT, ObjectType.WALL]
        for k in obstacle_keys:
            olst = semantic_map[k]
            if len(olst) > 0:
                obstacle_lst.extend(olst)
        return obstacle_lst

    def _registerObstacle(self, obstacle):
        """ Register a new obstacle in the tracker
        """
        self.obstacles[self.oid] = obstacle
        self.appeared[self.oid] = 1
        self.disappeared[self.oid] = 0
        self.oid += 1

    def _deregisterObstacle(self, oid):
        """ Deregister an obstacle from the tracker
        """
        del self.obstacles[oid]
        del self.appeared[oid]
        del self.disappeared[oid]

    def _computeDistance(self, o1, o2):
        """ Computes euclidean distance between two obstacles
        """
        return np.linalg.norm(o1['center'] - o2['center'])

    def _gating(self, assoc_mat):
        """ Ignore all associations whose cost is higher than gating_t threshold
        Based on Simone's localization heuristics
        """
        m = np.arange(0, assoc_mat.shape[0])
        n = np.argmin(assoc_mat, axis=1)
        a_mn = assoc_mat[m, n]
        # Observations that doesnt pass gatings are considered new obstacles
        indices_ko = np.where(a_mn >= self.gating_t)[0]
        new_indices = np.take(m, indices_ko, axis=0)
        # Save associations that pass the gating
        indices_ok = np.where(a_mn < self.gating_t)[0]
        m = np.take(m, indices_ok, axis=0)[..., None]
        n = np.take(n, indices_ok, axis=0)[..., None]
        a_mn = np.take(a_mn, indices_ok, axis=0)[..., None]
        associations = np.concatenate([m, n, a_mn], axis=-1)
        return associations, new_indices

    def _bestFriend(self, associations, assoc_mat):
        """ Keeps associations which are the best (i.e. minimum) of both row and column
        Based on Simone's localization heuristics
        """
        if associations.shape[0] == 0:
            return np.array([]), np.array([])
        # min by columns
        mm = np.argmin(assoc_mat, axis=0)
        # min by rows
        m, n = associations[:, 0].astype(int), associations[:, 1].astype(int)
        # Take those who are in n
        mm = np.take(mm, n)
        # Take those obs whose a_mn is a minimum of rows and cols
        indices = np.where(m==mm)[0]
        associations = np.take(associations, indices, axis=0)
        indices = np.where(m!=mm)[0]
        pruned = np.take(m, indices, axis=0)
        return associations, pruned

    def _lonelyBestFriend(self, associations, assoc_mat):
        """ One measurement should be only assigned to one single landmark
        Based on Simone's localization heuristics
        """
        if associations.shape[0] == 0:
            return np.array([]), np.array([])
        m, n, a_mn = associations[:, 0].astype(int), associations[:, 1].astype(int), associations[:, 2]
        assoc_mat = assoc_mat.copy()
        assoc_mat[m, n] = np.inf
        a_min_row = np.take(np.min(assoc_mat, axis=1), m)
        a_min_col = np.take(np.min(assoc_mat, axis=0), n)
        # Check wether current associations are in safe region w.r.t second minimum
        cond = np.logical_or(a_min_row - a_mn < self.lbf_t, a_min_col - a_mn < self.lbf_t)
        indices = np.where(np.logical_not(cond))[0]
        associations = np.take(associations, indices, axis=0)
        # Save observations that didn't pass third gating
        indices = np.where(cond)[0]
        pruned = np.take(m, indices, axis=0)
        return associations, pruned

    def process(self, semantic_map):
        """ Apply two-frame tracking.
        Each association is made based on the overlapping  
        """
        detected_obstacles = []
        remove_idx_lst = []
        current_obstacles = self._extractObstacles(semantic_map)
        # No obstacles found in the current frame
        if len(current_obstacles) == 0:
            # Loop over existing tracked obstacles and mark them as disappeared
            for k in self.obstacles.keys():
                self.disappeared[k] += 1
                # Remove obstacles who are disappeared for more than self.max_frames
                if self.disappeared[k] > self.max_frames:
                    remove_idx_lst.append(k)
            for k in remove_idx_lst:
                self._deregisterObstacle(k)
            return np.array([]), self.obstacles
        # No previous obstacles, but new ones found
        # If no obstacles are registered, register the current ones
        if len(self.obstacles) == 0:
            for obst in current_obstacles:
                self._registerObstacle(obst)
        # General case (Obstacles saved and new proposals are found)
        else:
            # Associate new obstacles with ones previously tracked and update/add
            # Generate association cost matrix
            nr = len(current_obstacles)
            nc = len(self.obstacles.keys())
            assoc_mat = np.zeros((nr, nc))
            obst_keys = list(self.obstacles.keys())
            # Fill assoc_mat
            for r in range(nr):
                for c in range(nc):
                    o1 = current_obstacles[r]
                    o2 = self.obstacles[obst_keys[c]]
                    assoc_mat[r, c] = np.linalg.norm(o1['center'] - o2['center'])
            # Apply gating
            associations, new_indices = self._gating(assoc_mat)
            associations, bf_pruned   = self._bestFriend(associations, assoc_mat)
            associations, lbf_pruned  = self._lonelyBestFriend(associations, assoc_mat)
            doubtful_indices = np.concatenate([bf_pruned, lbf_pruned])
            logger.debug(assoc_mat)
            logger.debug(f'new_indices:{new_indices}, bf_pruned:{bf_pruned}, lbf_pruned:{lbf_pruned}')
            logger.debug(f'associations:{associations}')

            # First apply decay to all registered objects
            # Then remove the decay for tracked ones
            for k in self.obstacles.keys():
                self.disappeared[k] += 1

            # Update obstacles whose associations were found
            # each association is in the form [row, col, distance]
            for assoc in associations:
                okey = obst_keys[assoc[1].astype(int)]
                self.obstacles[okey] = current_obstacles[assoc[0].astype(int)]
                #print(f'Association: id={okey}, center={self.obstacles[okey]["center"]}')
                self.appeared[okey] += 1
                # Remove decay
                self.disappeared[okey] = min(self.disappeared[okey]-1, 0)
            # Register new obstacles
            for new_idx in new_indices:
                self._registerObstacle(current_obstacles[new_idx])

            # Demote doubtful obstacles
            for k in self.obstacles.keys():
                if self.disappeared[k] > self.max_frames:
                    remove_idx_lst.append(k)
            for k in remove_idx_lst:
                self._deregisterObstacle(k)

        
        for k in self.obstacles.keys():
            if self.appeared[k] > self.min_frames:
                detected_obstacles.append(self.obstacles[k])

        detected_obstacles = self._computeObstacleFeatures(detected_obstacles)        
        return detected_obstacles, self.obstacles

    def _computeObstacleFeatures(self, obstacles):
        """ Computes features for each tracked obstacle.
        Features include:
        - Distance vector from bottom-center point to obstacle
        - Estimated radius of the obstacle
        - Inside-lane flag (TODO)
        """
        BASELINE_POINT = np.int16([320, 480]) # Lower center camera pixel

        def computeDistanceRad(x, y, w, h):
            end_pt = np.array([x + w/2, y+h])
            radius = w/2
            distance_vect = end_pt - BASELINE_POINT
            return distance_vect, radius
        
        for object in obstacles:
            x, y, w, h = cv2.boundingRect(object['contour']) # Bounding box
            distance_vect, radius = computeDistanceRad(x, y, w, h)
            object['bbox'] = (x, y, w, h)
            object['end_point'] = np.array([x + w/2, y+h])
            object['end_right'] = np.array([x, y+h])
            object['end_lat'] = np.array([x, y+h/2])
            object['end_top'] = np.array([x, y])
            object['distance'] = distance_vect
            object['radius'] = radius
            # TODO
            object['in_lane'] = True

        return obstacles

