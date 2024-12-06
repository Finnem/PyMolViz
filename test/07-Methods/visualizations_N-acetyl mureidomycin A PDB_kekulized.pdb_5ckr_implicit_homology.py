
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 1")

cmd.set("stick_transparency", 0.5797539495188044, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 2")

cmd.set("stick_transparency", 0.6061096686461376, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 3")

cmd.set("stick_transparency", 0.6073843590209154, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 4")

cmd.set("stick_transparency", 0.6918751677496895, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 5")

cmd.set("stick_transparency", 0.7814463178801252, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 6")

cmd.set("stick_transparency", 0.8030123913777973, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 7")

cmd.set("stick_transparency", 0.8113201293370658, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 8")

cmd.set("stick_transparency", 0.8427844865638014, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 9")

cmd.set("stick_transparency", 0.8802724043400115, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 10")

cmd.set("stick_transparency", 0.8893918090496074, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 11")

cmd.set("stick_transparency", 0.8960669227709058, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 12")

cmd.set("stick_transparency", 0.9055920842780368, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 13")

cmd.set("stick_transparency", 0.917075542105523, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 14")

cmd.set("stick_transparency", 0.9173123441345612, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 15")

cmd.set("stick_transparency", 0.9182553937042726, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 16")

cmd.set("stick_transparency", 0.9207724063811755, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 17")

cmd.set("stick_transparency", 0.9229040296256316, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 18")

cmd.set("stick_transparency", 0.9284303759328512, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 19")

cmd.set("stick_transparency", 0.9303355142286736, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 20")

cmd.set("stick_transparency", 0.9335242307826701, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 21")

cmd.set("stick_transparency", 0.9339795207146605, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 22")

cmd.set("stick_transparency", 0.9351060495316031, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 23")

cmd.set("stick_transparency", 0.9376729073901685, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 24")

cmd.set("stick_transparency", 0.9399107498860957, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 25")

cmd.set("stick_transparency", 0.9467527212010949, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 26")

cmd.set("stick_transparency", 0.9468048639051063, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 27")

cmd.set("stick_transparency", 0.9477099484390067, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 28")

cmd.set("stick_transparency", 0.9486114165032309, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 29")

cmd.set("stick_transparency", 0.9492679008882063, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 30")

cmd.set("stick_transparency", 0.95, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr_implicit_homology and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
