from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements

    Parameters
    ----------
    file_path : str
        this is the path of the file where all the requirements are specified

    Returns
    -------
    List[str]
        list of of the packages mentioned in the requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="ml_project_hands_on",
    version="0.0.1",
    description="hands on project end to end development",
    author="D J H",
    author_email="djhiremath@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
