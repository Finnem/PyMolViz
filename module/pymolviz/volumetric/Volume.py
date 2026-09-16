import numpy as np
import logging
from .GridData import GridData
from ..Displayable import Displayable
from ..ColorMap import ColorMap
from ..util.colors import _convert_string_color

VOLUME_RAMP_SAMPLES = 33
_BIN_COUNT_CHUNK = 262144


def _expand_range_to_volume_clims(min_val, max_val, samples=VOLUME_RAMP_SAMPLES):
    """Paired bin edges so density alphas and colormap RGB are sampled densely."""
    lo, hi = float(min_val), float(max_val)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-15:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, max(2, int(samples)))
    paired = np.vstack([edges[:-1], edges[1:]]).T.flatten()
    return np.hstack([paired, paired[-1]])


def _pair_bin_counts(values, bins):
    """Count values in each ``(lo, hi]`` pair without an ``(N, n_bins)`` broadcast."""
    lo = np.asarray(bins[:, 0], dtype=float)
    hi = np.asarray(bins[:, 1], dtype=float)
    n_bins = int(lo.size)
    counts = np.zeros(n_bins, dtype=np.int64)
    vals = np.asarray(values, dtype=float).reshape(-1)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0 or n_bins == 0:
        return counts
    gap = 1e-9 * (1.0 + np.abs(hi[:-1]))
    contiguous = n_bins == 1 or np.all(np.abs(lo[1:] - hi[:-1]) <= gap)
    if contiguous:
        edges = np.concatenate([lo, hi[-1:]])
        idx = np.searchsorted(edges, finite, side="left") - 1
        valid = (idx >= 0) & (idx < n_bins)
        np.add.at(counts, idx[valid], 1)
        return counts
    for start in range(0, finite.size, _BIN_COUNT_CHUNK):
        sl = finite[start:start + _BIN_COUNT_CHUNK]
        for i in range(n_bins):
            counts[i] += np.count_nonzero((sl > lo[i]) & (sl <= hi[i]))
    return counts


def _rgb_from_colormap(colormap, value):
    rgba = np.asarray(colormap.get_color(value), dtype=float).reshape(-1)
    return [float(rgba[0]), float(rgba[1]), float(rgba[2])]


def _volume_object_names(cmd):
    try:
        return [str(n) for n in cmd.get_names("objects")]
    except Exception:
        return []


def _apply_volume_ramp(cmd, volume_name, ramp_name, flat_list):
    volume_color = getattr(cmd, "volume_color", None)
    if not callable(volume_color):
        return False
    for argument in (ramp_name, flat_list):
        try:
            volume_color(volume_name, argument)
            return True
        except TypeError:
            continue
        except Exception:
            return False
    return False


class Volume(Displayable):
    renders_cgo = False

    def __init__(self, grid_data : GridData, name = None, colormap = "RdYlBu_r", alphas = None, clims = None, selection = None, carve = None, state = 1, use_min_max = False, geometry_field_id=None, color_field_id=None, clip_aabb=None, transfer_stops=None):
        """ 
        Computes and collects pymol commands to load in regular data and display it volumetrically.

        Args:
            grid_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {grid_data.name}_{value_label}_Volume_{i}.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.03, 0.005, 0.1].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
            state (int, optional): The state to use. Defaults to 1.
        """

        self.grid_data = grid_data
        self.selection = selection
        self.carve = carve
        self.state = state
        self.is_loaded = False
        self.A_to = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.geometry_field_id = str(geometry_field_id) if geometry_field_id else None
        self.color_field_id = str(color_field_id) if color_field_id else None
        from ..fields.clip import normalize_clip_aabb
        self.clip_aabb = normalize_clip_aabb(clip_aabb)

        super().__init__(name = name)
        
        if clims is None:
            if use_min_max:
                min_val = np.min(grid_data.values)
                max_val = np.max(grid_data.values)
            else:
                min_val = max([np.min(grid_data.values), -np.std(grid_data.values) * 5 + np.mean(grid_data.values)])
                max_val = min([np.max(grid_data.values), np.std(grid_data.values) * 5 + np.mean(grid_data.values)])
            self.clims = _expand_range_to_volume_clims(min_val, max_val)
        else:
            clim_arr = np.asarray(clims, dtype=float).reshape(-1)
            # Wizard range is [vmin, vmax]; a two-knot ramp would ignore middle colormap stops.
            if clim_arr.size == 2 and alphas is None:
                self.clims = _expand_range_to_volume_clims(clim_arr[0], clim_arr[1])
            else:
                self.clims = clim_arr

        if not issubclass(type(colormap), ColorMap):
            colormap = ColorMap(self.clims, colormap, state = state, name = f"{self.name}_colormap")
        self.colormap = colormap

        if alphas is None:
            used_length = len(self.clims)-(len(self.clims) % 2) # if length is uneven, we forgo the last value for binning
            bins = np.reshape(self.clims[:used_length], (-1, 2))
            densities = _pair_bin_counts(self.grid_data.values, bins).astype(float)
            if np.sum(densities) == 0: densities = np.ones(len(densities))
            densities = densities / np.sum(densities)
            densities = np.clip(densities, None, 0.9)
            self.alphas = np.vstack([(1 - densities[:used_length]) * 0.03, np.full(len(densities[:used_length]), 0.005)]).T.flatten()
            if len(self.clims) % 2 == 1:
                self.alphas = np.hstack([self.alphas, (1 - densities[-1]) * 0.03])
        else:
            self.alphas = np.array(alphas)
        if len(self.alphas) != len(self.clims):
            raise Exception("Alphas and clims must have the same length.")

        if transfer_stops is not None:
            from ..fields.isovalues import normalize_transfer_stops
            self.transfer_stops = normalize_transfer_stops(transfer_stops)
        else:
            from ..fields.isovalues import stops_from_volume_ramp
            self.transfer_stops = stops_from_volume_ramp(self.clims, self.alphas)
        
        self.dependencies.extend([self.grid_data])


    def cut(self, point : np.array, normal : np.array,interpolation = "NN"):
        """
        Cut the volume along a plane defined by a point on the plane and a normal. 
        
        Args:
            point (np.array): point on the plane
            normal (np.array): normal of the plane
            interpolation (np.array, optional): Defaults to "NN". The interpolation method to use. Can be "NN" (nearest neighbor) or "Lin/NN" linear and nearest neighbor. 
        """
        normal = np.array(normal/np.linalg.norm(normal))
        point = np.array(point)

        new_grid_data = self.grid_data.cut(point, normal, interpolation)
        
        new_volume = type(self)(new_grid_data, name="%s_cut"%self.name, alphas = self.alphas, clims = self.clims, colormap = self.colormap, state = self.state)
        new_volume.A_to = new_volume.grid_data.A_to

        if self.is_loaded:
            new_volume.load()
        return new_volume
        

    def _script_string(self):
        """ Creates a pymol script to create a volume representation of the given regular data.
        
        Returns:
            str: The script.
        """
        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")

        string_list = []


        if len(self.alphas) != len(self.clims):
                raise ValueError("The number of volume alphas must be equal to the number of clims.")
        string_list = [f"""cmd.volume_ramp_new("{self.name}_volume_color_ramp", [\\"""]
        for i, c in enumerate(self.clims):
            string_list.append(f"""    {self.clims[i]}, {",".join([str(v) for v in _rgb_from_colormap(self.colormap, c)])}, {self.alphas[i]},\\""")
        string_list.append("])")
        string_list.append(f"""
cmd.volume("{self.name}", "{self.grid_data.name}", "{self.name}_volume_color_ramp", {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""} state={self.state})
cmd.set("volume_mode", 0)
        """)

        result = "\n".join(string_list)
        return result

    def load(self, cmd=None):
        if cmd is None:
            from pymol import cmd
        
        if len(self.alphas) != len(self.clims):
                raise ValueError("The number of volume alphas must be equal to the number of clims.")
        params_list = [[self.clims[i], _rgb_from_colormap(self.colormap, c), self.alphas[i]] for i, c in enumerate(self.clims)]
        flat_list = []
        for sublist in params_list:
            for item in sublist:
                if type(item) == list:
                    for list_item in item:
                        flat_list.append(float(list_item))
                else:
                    flat_list.append(float(item))
        from .map_load import load_geometry_map, sync_visual_grid_from_field

        sync_visual_grid_from_field(self, cmd)
        map_name, rebuilt = load_geometry_map(cmd, self.grid_data, self.clip_aabb)
        if not map_name:
            return
        ramp_name = "%s_volume_color_ramp" % self.name
        cmd.volume_ramp_new(ramp_name, flat_list)
        exists = self.name in _volume_object_names(cmd)
        if exists and not rebuilt:
            state_ok = True
            getter = getattr(cmd, "get_object_state", None)
            if callable(getter):
                try:
                    state_ok = int(getter(self.name)) == int(self.state)
                except Exception:
                    state_ok = True
            if state_ok and _apply_volume_ramp(cmd, self.name, ramp_name, flat_list):
                cmd.set("volume_mode", 0)
                self.is_loaded = True
                return
        if exists:
            try:
                cmd.delete(self.name)
            except Exception:
                pass
            exists = self.name in _volume_object_names(cmd)
        if exists:
            logging.warning(
                "The volume could not be updated because %s (state %s) already exists."
                % (self.name, self.state)
            )
            return
        volume_kwargs = {"ramp": ramp_name, "state": self.state}
        if self.selection:
            volume_kwargs["selection"] = self.selection
        if self.carve is not None:
            volume_kwargs["carve"] = self.carve
        cmd.volume(self.name, map_name, **volume_kwargs)
        cmd.set("volume_mode", 0)
        self.is_loaded = True

    def to_script(self, state = 0):
        """ Creates a pymolviz script to create a volume representation of the given regular data.
        
        Returns:
            pymolviz.Script: The script.
        """
        from ..Script import Script
        return Script([self])
