class Board:
    # Encapsulates board size, cell size, and bounds checks.
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_cell_size(self):
        return self.cell_size