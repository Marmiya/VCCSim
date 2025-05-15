import sys
import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Add the parent directory of PyClient to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient
from VCCSim import VCCSim_pb2
from VCCSim.VCCSimClient import RGBImageUtils

def test_recording():
    # Create VCCSim client with default connection settings
    client = VCCSimClient(host="172.31.178.18", port=50996)
    client.toggle_recording()
    

if __name__ == "__main__":
    
    test_recording()