import argparse
import os, sys
from json.encoder import INFINITY
from src.parser import upperParser

p = os.path.abspath('.')
sys.path.insert(1, p)

def main():
    parser = upperParser()

main()
