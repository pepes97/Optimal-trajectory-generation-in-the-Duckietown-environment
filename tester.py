"""tester.py
Launcher for tests on the repository
"""


import logging
import argparse
from lib.tests import *
from lib.utils import bcolors

logger=logging.getLogger(__name__)

VALID_TEST_LST = [
    'path_follower_2D',
    'trajectory_track_2D',
    'simlogger'
]

TEST_MAP = {
    'path_follower_2D' : test_path_follower_2D,
    'trajectory_track_2D' : test_trajectory_track_2D,
    'simlogger' : test_simlogger
}

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
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {loglevel}')
    logging.basicConfig(level=numeric_level)
        

if __name__ == '__main__':
    # CLI arg parser
    parser = argparse.ArgumentParser(description='Repository tester.')
    parser.add_argument('--test', metavar='t', type=str, help='test name', required=True)
    parser.add_argument('--log', metavar='l', type=str, help='logging level', default='WARNING')
    args = parser.parse_args()
    handle_parser(args)

    # Execute test routine
    print(f'{bcolors.OKGREEN}Launching test for {args.test}{bcolors.ENDC}')
    result = TEST_MAP[args.test]()
    if result is not None:
        # Process results
        pass
    
    exit(0)
