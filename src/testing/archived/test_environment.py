import os.path
import sys

sys.path.insert(0, os.path.abspath('../'))
import dm


def test_env_init_local():

    assert dm.env.source is 'local'
