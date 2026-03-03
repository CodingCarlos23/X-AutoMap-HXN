class AppState:
    """A class to hold the application state."""
    def __init__(self):
        self.file_paths = None
        self.selected_files_order = None
        self.img_paths = [None, None, None]
        self.file_names = [None, None, None]
        self.true_origin_x = 0.0
        self.true_origin_y = 0.0
        self.target_shape = None
        self.element_colors = []
        self.thresholds = {}
        self.area_thresholds = {}
        self.precomputed_blobs = {
            "red": {},
            "green": {},
            "blue": {}
        }
        self.microns_per_pixel_x = 1.0
        self.microns_per_pixel_y = 1.0
        self.selected_directory = None
