import shutil
import configparser
from pathlib import Path, PosixPath
from setuptools import find_packages, setup

PACKAGE_NAME = 'datatools'

_CWD = Path(__file__).parent
_DIST_DIR = _CWD / 'dist'
_BUILD_DIR = _CWD / 'build'
_EGG_INFO_DIR = _CWD / f'{PACKAGE_NAME}.egg-info'

def get_version():
    config = configparser.ConfigParser()
    config.read(_CWD / PACKAGE_NAME / 'version.cfg')
    return config.get("version", "update")

def clean_dir(dir: PosixPath):
    if dir.exists():
        shutil.rmtree(dir)

clean_dir(_DIST_DIR)
clean_dir(_BUILD_DIR)
clean_dir(_EGG_INFO_DIR)

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    author='Jianyao Sun',
    author_email='jianyao.sun@foxmail.com',
    packages=find_packages(
        exclude=(
            'test',
            'demo',
            'build',
            'dist',
            '__pycache__'
            f'{PACKAGE_NAME}.egg-info'
        )
    ),
    package_data={
        PACKAGE_NAME: ['*.cfg']
    },
    exclude_package_data={
        PACKAGE_NAME: ['*.ipynb']
    },

)