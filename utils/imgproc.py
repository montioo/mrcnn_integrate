import attr

""" The default pixel coordinate type
"""
@attr.s
class PixelCoord:
    x: int = -1
    y: int = -1

    @property
    def row(self):
        return self.y

    @property
    def col(self):
        return self.x