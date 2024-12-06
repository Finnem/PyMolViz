
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 1")

cmd.set("stick_transparency", 0.06441002419327935, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 2")

cmd.set("stick_transparency", 0.4212791861827936, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 3")

cmd.set("stick_transparency", 0.5328092234309598, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 4")

cmd.set("stick_transparency", 0.8027433448449091, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 5")

cmd.set("stick_transparency", 0.8363318426468513, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 6")

cmd.set("stick_transparency", 0.8625226891565515, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 7")

cmd.set("stick_transparency", 0.8863864271965068, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 8")

cmd.set("stick_transparency", 0.8881237349526391, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 9")

cmd.set("stick_transparency", 0.8896621785551905, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 10")

cmd.set("stick_transparency", 0.8924977536758041, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 11")

cmd.set("stick_transparency", 0.9060312881175024, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 12")

cmd.set("stick_transparency", 0.9061475058065653, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 13")

cmd.set("stick_transparency", 0.9177262071415033, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 14")

cmd.set("stick_transparency", 0.9191579689980907, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 15")

cmd.set("stick_transparency", 0.9198708928093291, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 16")

cmd.set("stick_transparency", 0.9280547714733396, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 17")

cmd.set("stick_transparency", 0.928915209734166, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 18")

cmd.set("stick_transparency", 0.930344078864997, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 19")

cmd.set("stick_transparency", 0.9348130818968704, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 20")

cmd.set("stick_transparency", 0.9350391575764373, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 21")

cmd.set("stick_transparency", 0.9358843942330636, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 22")

cmd.set("stick_transparency", 0.9394731273189052, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 23")

cmd.set("stick_transparency", 0.9394782081126293, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 24")

cmd.set("stick_transparency", 0.941095427367467, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 25")

cmd.set("stick_transparency", 0.9447826781420754, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 26")

cmd.set("stick_transparency", 0.9467977316719265, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 27")

cmd.set("stick_transparency", 0.9473256301236439, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 28")

cmd.set("stick_transparency", 0.9475917781782702, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 29")

cmd.set("stick_transparency", 0.9485753248835401, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 30")

cmd.set("stick_transparency", 0.95, f"Tunicamycin_PDB_kekulized.pdb_alpha_fold2_Visualizations and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
