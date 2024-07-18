import os

# pip >=20
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

REQ_FOLDER = 'requirements'
REQ_FILE = 'requirements.txt'
TEST_REQ_FILE = 'requirements.txt'
DEV_REQ_FILE = 'requirements.txt'


def on_ci():
    return os.getenv("CI_JOB_STAGE") is not None


def is_testing():
    return os.getenv("CI_JOB_STAGE") == 'test'


def get_requirement_file_name() -> str:
    file_name = DEV_REQ_FILE
    if is_testing() or not on_ci() and not os.path.isfile(os.path.join(os.path.dirname(__file__), REQ_FOLDER, DEV_REQ_FILE)):
        file_name = TEST_REQ_FILE
    elif on_ci():
        file_name = REQ_FILE

    return file_name


def get_requirement_path() -> str:
    return os.path.join(os.path.dirname(__file__), REQ_FOLDER, get_requirement_file_name())


def get_req():
    install_reqs = parse_requirements(get_requirement_path(), session=PipSession())
    reqs = [str(ir.requirement) for ir in install_reqs]
    return reqs
