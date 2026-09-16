from .ColorRamp import ColorRamp
from .IsoSurface import IsoSurface

class IsoMesh(IsoSurface):
    _native_iso_cmd = "isomesh"
    _native_iso_passes_side = False
