import io
import os
import random
import re
import sys
import xml.etree.ElementTree

import yaml
from train_PointNet_car import *

# reads in parameters from params.yaml
params = yaml.safe_load(open(os.path.join(HOME_PATH, "params.yaml")))["train"]
