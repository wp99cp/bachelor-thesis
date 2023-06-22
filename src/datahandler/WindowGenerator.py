from rasterio.windows import from_bounds

WINDOWS = {
    '32TNS': {
        "left": 499980,
        "bottom": 5090220,
        "right": 609780,
        "top": 5200020
    },
    '32VMP': {
        "left": 400050,
        "bottom": 6790330,
        "right": 509740,
        "top": 6899980
    },
    '13TDE': {
        "left": 400260,
        "bottom": 4390510,
        "right": 509720,
        "top": 4499940
    },
    '07VEH': {
        "left": 501290,
        "bottom": 6691290,
        "right": 608330,
        "top": 6798330
    }
}


class WindowGenerator:

    def __init__(self, transform):
        self.transform = transform

    def get_window(self, tile_id: str):
        assert tile_id in WINDOWS.keys(), f"Tile id {tile_id} not found in WINDOWS dictionary."
        window = WINDOWS[tile_id]
        return from_bounds(left=window["left"],
                           bottom=window["bottom"],
                           right=window["right"],
                           top=window["top"],
                           transform=self.transform)
