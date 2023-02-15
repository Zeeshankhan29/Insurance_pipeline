from pathlib import Path
import os
from src.Insurance import logger


# lg.basicConfig(filename='Insurance/logger/template.log',level=lg.DEBUG, format='[%(asctime)s] : [%(levelname)s] : [%(message)s]')
# stream = lg.StreamHandler()
# log = lg.getLogger()
# log.addHandler(stream)

while True:
    project_name = input('Enter the project name  :  ')
    if project_name !='':
        break

logger.info(f'Project {project_name} creation started')


list_of_files =[

    '.github/workflows/.gitkeep',
    '.github/workflows/main.yaml',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/pipeline/__init__.py',
    'configs/config.yaml',
    'requirements.txt',
    'setup.py',
    'main.py',
    'mongo_dump.py',

    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !='':
        os.makedirs(filedir,exist_ok=True)
        logger.info(f'creating directory {filedir} for file {filename}')
    
    if not os.path.exists(filepath) or os.path.getsize(filepath)==0:
        with open(filepath,'w') as f:
            pass
        logger.info(f'Creating a empty {filename}')
    else:
        logger.info(f'file {filename} already exists')