
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 1")

cmd.set("stick_transparency", 0.21215275829882785, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 2")

cmd.set("stick_transparency", 0.7084588664737527, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 3")

cmd.set("stick_transparency", 0.7439052115274635, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 4")

cmd.set("stick_transparency", 0.7863937762777458, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 5")

cmd.set("stick_transparency", 0.7979026376685258, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 6")

cmd.set("stick_transparency", 0.8265274364168436, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 7")

cmd.set("stick_transparency", 0.8374430668350901, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 8")

cmd.set("stick_transparency", 0.850874916055781, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 9")

cmd.set("stick_transparency", 0.8510272282421181, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 10")

cmd.set("stick_transparency", 0.8591523030040806, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 11")

cmd.set("stick_transparency", 0.8698514239539799, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 12")

cmd.set("stick_transparency", 0.8705435436180988, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 13")

cmd.set("stick_transparency", 0.8812388215725184, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 14")

cmd.set("stick_transparency", 0.8898417075384811, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 15")

cmd.set("stick_transparency", 0.8927868642913717, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 16")

cmd.set("stick_transparency", 0.8932750310504752, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 17")

cmd.set("stick_transparency", 0.8954263810399226, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 18")

cmd.set("stick_transparency", 0.8971741015485668, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 19")

cmd.set("stick_transparency", 0.8988696211208473, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 20")

cmd.set("stick_transparency", 0.9059472801898878, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 21")

cmd.set("stick_transparency", 0.9108774853886659, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 22")

cmd.set("stick_transparency", 0.9116149954114203, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 23")

cmd.set("stick_transparency", 0.9193652798923835, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 24")

cmd.set("stick_transparency", 0.9244990975393151, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 25")

cmd.set("stick_transparency", 0.9255825288709728, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 26")

cmd.set("stick_transparency", 0.9308636207351535, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 27")

cmd.set("stick_transparency", 0.9334746259649311, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 28")

cmd.set("stick_transparency", 0.9419322876884586, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 29")

cmd.set("stick_transparency", 0.9483833326645866, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 30")

cmd.set("stick_transparency", 0.95, f"Tunicamycin_PDB_kekulized.pdb_5ckr_implicit_homology and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
