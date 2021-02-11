"""tester.py
Launcher for tests on the repository
"""


import logging
import argparse
import datetime
import os
import errno
from matplotlib import pyplot as plt
from lib.tests import *
from lib.utils import bcolors
from lib.plotter import plot_2d_simulation


logger=logging.getLogger(__name__)

TEST_MAP = {
    'path_follower_2D' : test_path_follower_2D,
    'trajectory_track_2D' : test_trajectory_track_2D,
    'simlogger' : test_simlogger,
    'serializer' : test_serializer,
}

TEST_PRINT_MAP = {
    'path_follower_2D' : plot_2d_simulation,
    'trajectory_track_2D' : plot_2d_simulation,
    'simlogger' : None,
    'serializer' : None,
}

IMG_PATH = './images/'

def handle_parser(args):
    if args.test not in TEST_MAP.keys():
        print(f'{bcolors.FAIL}The insterted test is not valid.{bcolors.ENDC}')
        print(f'Available tests are:')
        for tstr in TEST_MAP.keys():
            print(tstr)
        exit(0)
    # Setup logger
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper())
    # Disable matplotlib logger
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {loglevel}')
    logging.basicConfig(level=numeric_level)
        

if __name__ == '__main__':
    # CLI arg parser
    parser = argparse.ArgumentParser(description='Repository tester.')
    parser.add_argument('--test', '-t', metavar='t', type=str, help='test name', required=True)
    parser.add_argument('--log', '-l',  metavar='l', type=str, help='logging level', default='WARNING')
    parser.add_argument('--store', '-s',  metavar='s', type=str, help='log path')
    parser.add_argument('--print', '-p', help='Print flag', action='store_true')
    args = parser.parse_args()
    handle_parser(args)

    # Execute test routine
    print(f'{bcolors.OKGREEN}Launching test for {args.test}{bcolors.ENDC}')
    result = TEST_MAP[args.test]()
    if result is not None:
        # Process results
        # Store results
        if args.store is not None:
            log_path = args.store
            logger.warning('Store and load callbacks are not ready yet.')
            #print(f'{bcolors.OKGREEN}Storing results to {log_path}{bcolors.ENDC}')
            #result.save(log_path)
        if args.print is True and TEST_PRINT_MAP[args.test] is not None:
            print(f'{bcolors.OKGREEN}Printing results{bcolors.ENDC}')
            result_figure = TEST_PRINT_MAP[args.test](result)
            plt.show()
            # Store plot
            res_dir = os.path.join(os.path.join(os.getcwd(), IMG_PATH), datetime.datetime.now().strftime('%Y%m%d'))
            try:
                os.makedirs(res_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise RuntimeError
            fig_path = os.path.join(res_dir, f'{args.test}.jpg')
            print(f'{bcolors.OKGREEN}Storing plot in :{bcolors.ENDC}{fig_path} ')
            result_figure.savefig(fig_path)
    
    exit(0)
