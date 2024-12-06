
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 1")

cmd.set("stick_transparency", 0.7714166412736541, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 2")

cmd.set("stick_transparency", 0.8228685117842276, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 3")

cmd.set("stick_transparency", 0.8535984153080618, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 4")

cmd.set("stick_transparency", 0.8685256621290245, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 5")

cmd.set("stick_transparency", 0.8877320148423459, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 6")

cmd.set("stick_transparency", 0.9014672352568975, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 7")

cmd.set("stick_transparency", 0.9044982481373438, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 8")

cmd.set("stick_transparency", 0.9048389828977594, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 9")

cmd.set("stick_transparency", 0.9095033060658155, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 10")

cmd.set("stick_transparency", 0.917565427310758, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 11")

cmd.set("stick_transparency", 0.9189887561614948, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 12")

cmd.set("stick_transparency", 0.9255211264573168, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 13")

cmd.set("stick_transparency", 0.9261204195342829, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 14")

cmd.set("stick_transparency", 0.9297162170739142, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 15")

cmd.set("stick_transparency", 0.9311930762784092, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 16")

cmd.set("stick_transparency", 0.9319951276693036, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 17")

cmd.set("stick_transparency", 0.9349531010733656, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 18")

cmd.set("stick_transparency", 0.9354162255467836, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 19")

cmd.set("stick_transparency", 0.9402314925024406, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 20")

cmd.set("stick_transparency", 0.9406989874359394, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 21")

cmd.set("stick_transparency", 0.9409041780717227, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 22")

cmd.set("stick_transparency", 0.941843443550217, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 23")

cmd.set("stick_transparency", 0.9439367265270394, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 24")

cmd.set("stick_transparency", 0.9441766865234676, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 25")

cmd.set("stick_transparency", 0.9445615782046876, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 26")

cmd.set("stick_transparency", 0.9471561901807873, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 27")

cmd.set("stick_transparency", 0.9479704101486472, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 28")

cmd.set("stick_transparency", 0.9483267076848098, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 29")

cmd.set("stick_transparency", 0.9484271868896388, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 30")

cmd.set("stick_transparency", 0.95, f"Tunicamycin_PDB_kekulized.pdb_5ckr and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
