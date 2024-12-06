
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 1")

cmd.set("stick_transparency", 0.42358953568329993, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 2")

cmd.set("stick_transparency", 0.5628429806217213, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 3")

cmd.set("stick_transparency", 0.6719820881260581, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 4")

cmd.set("stick_transparency", 0.7080220228755616, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 5")

cmd.set("stick_transparency", 0.7592014050546153, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 6")

cmd.set("stick_transparency", 0.8144815136365195, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 7")

cmd.set("stick_transparency", 0.8257241024575981, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 8")

cmd.set("stick_transparency", 0.841904521212824, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 9")

cmd.set("stick_transparency", 0.8833955007819045, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 10")

cmd.set("stick_transparency", 0.8967900254730583, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 11")

cmd.set("stick_transparency", 0.8985998209567031, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 12")

cmd.set("stick_transparency", 0.910100476133316, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 13")

cmd.set("stick_transparency", 0.9133589011685127, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 14")

cmd.set("stick_transparency", 0.9137037570895048, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 15")

cmd.set("stick_transparency", 0.9147899193262404, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 16")

cmd.set("stick_transparency", 0.9234494644767216, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 17")

cmd.set("stick_transparency", 0.9278119605244086, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 18")

cmd.set("stick_transparency", 0.9289192017478093, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 19")

cmd.set("stick_transparency", 0.9318044873773456, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 20")

cmd.set("stick_transparency", 0.935336542047591, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 21")

cmd.set("stick_transparency", 0.9370091650520332, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 22")

cmd.set("stick_transparency", 0.9406462023727546, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 23")

cmd.set("stick_transparency", 0.9458878579928807, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 24")

cmd.set("stick_transparency", 0.945965279171519, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 25")

cmd.set("stick_transparency", 0.9461654519101569, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 26")

cmd.set("stick_transparency", 0.9469378260475683, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 27")

cmd.set("stick_transparency", 0.9471926816374715, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 28")

cmd.set("stick_transparency", 0.9482566207792356, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 29")

cmd.set("stick_transparency", 0.9497375537723839, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 30")

cmd.set("stick_transparency", 0.95, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_5ckr and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
