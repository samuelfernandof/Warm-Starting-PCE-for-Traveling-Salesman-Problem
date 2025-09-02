import os

class Config:
    def __init__(self):
        self.QXToken = os.getenv('QXToken', 'o15gfM7FHzTGZc9YO8dFwFop4QrUSwKaF6Z0SXPDUK-Y')
        self.SIMULATION = os.getenv('SIMULATION', 'True')