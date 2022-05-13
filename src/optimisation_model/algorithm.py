import pandas as pd

class Algorithm:
    def __init__(
            self,
    ):
        self.LENGTH_PER_BLOCK = 8*60
        self.TYPES_TRANS = ["CARD", "GASTRO", "GYN", "MED", "ORTH", "URO"]
        self.TYPES = set([j for j in range(0, len(self.TYPES_TRANS))])
        self.SURGERIES = []
        self.SURGERY_TYPES = []
        self.BLOCKS = []
        self.TYPE_PER_BLOCK = []
        self.DAY_PER_BLOCK = []
        self.ROOM_PER_BLOCK = []

    def read_input_blocks(self, csv_blocks):
        blocks_input = pd.read_csv(csv_blocks, sep=";")
        self.BLOCKS = [int(b) for b in blocks_input["BLOCK"].tolist()]
        self.TYPE_PER_BLOCK = blocks_input["TYPE"].tolist()
        self.DAY_PER_BLOCK = blocks_input["DAY"].tolist()
        self.ROOM_PER_BLOCK = blocks_input["ROOM"].tolist()

    
   
