"""simulation_logger.py
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimData:
    robot_pose = ('robot_pose', 3)
    robot_frenet_pose = ('robot_frenet_pose', 3)
    control    = ('control', 2)
    trajectory_2d = ('trajectory', 2)
    target_pos = ('target_pos', 2)
    point = ('point', 2)
    

class SimulationDataStorage:
    """ Container for simulation data gathered throughout experiments
    """
    def __init__(self, t: np.array):
        """ Setup the storage.
        t : (LEN_SIMULATION,) np.array
        """
        assert t is not None
        self.t = t
        self.sim_length = t.shape[0]
        self.db = {}

    def add_argument(self, *args, **kwds):
        if len(args) == 1 and isinstance(args[0], tuple):
            arg_tuple = args[0]
            self.add_argument(arg_tuple[0], arg_tuple[1])
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            id = args[0]
            dim = args[1]
            assert id not in self.db
            assert dim > 0
            logger.debug(f'Adding new entry {id} of dimensions ({dim}, {self.sim_length})')
            if dim == 1:
                self.db[id] = np.zeros(self.sim_length)
            else:
                self.db[id] = np.zeros((dim, self.sim_length))
    
    def set(self, id: str, data, idx: int):
        """ Set the idx-th value of id's storage
        """
        assert idx >= 0 and idx < self.sim_length
        if id not in self.db:
            logger.error(f'{id} is not a valid key. You should first call add_argument and insert the element type.')
            raise RuntimeError('Invalid ID')

        container = self.db[id]
        if container.shape == self.t.shape:
            container[idx] = data
        else:
            container[:, idx] = data

    def get(self, id: str) -> np.array:
        """ Returns the storage of id
        """
        if id not in self.db:
            return None
        return self.db[id]

    def get_i(self, id: str, idx: int):
        """ Returns the idx-th value of id's storage
        """
        if id not in self.db:
            return None
        container = self.db[id]
        if container.shape == self.t.shape:
            return container[idx]
        else:
            return container[:, idx]

    def load(self, path: str):
        """ Loads data from a file
        """
        # TODO
        logger.error('Function not implemented')
        pass

    def save(self, path: str):
        """ Save data to a file
        """
        # TODO
        logger.error('Function not implemented')
        pass
