
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cmd.set("stick_transparency", 0.0, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 1")

cmd.set("stick_transparency", 0.4699455494197149, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 2")

cmd.set("stick_transparency", 0.49596898266956246, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 3")

cmd.set("stick_transparency", 0.5355259879685557, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 4")

cmd.set("stick_transparency", 0.6132786153191665, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 5")

cmd.set("stick_transparency", 0.6570185997836692, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 6")

cmd.set("stick_transparency", 0.6862865674409427, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 7")

cmd.set("stick_transparency", 0.7234719137016178, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 8")

cmd.set("stick_transparency", 0.7383285954845477, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 9")

cmd.set("stick_transparency", 0.7463737116645662, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 10")

cmd.set("stick_transparency", 0.7872494247772733, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 11")

cmd.set("stick_transparency", 0.8508494127311613, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 12")

cmd.set("stick_transparency", 0.8760726766689112, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 13")

cmd.set("stick_transparency", 0.8823914270223777, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 14")

cmd.set("stick_transparency", 0.8833938180417392, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 15")

cmd.set("stick_transparency", 0.8929548040580423, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 16")

cmd.set("stick_transparency", 0.9005161235594369, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 17")

cmd.set("stick_transparency", 0.9143043315032253, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 18")

cmd.set("stick_transparency", 0.9223063634710762, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 19")

cmd.set("stick_transparency", 0.9223819726418898, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 20")

cmd.set("stick_transparency", 0.9251983776974504, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 21")

cmd.set("stick_transparency", 0.9258485978567497, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 22")

cmd.set("stick_transparency", 0.9279183626125356, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 23")

cmd.set("stick_transparency", 0.9291706755332229, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 24")

cmd.set("stick_transparency", 0.9311928622291937, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 25")

cmd.set("stick_transparency", 0.9342930944191674, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 26")

cmd.set("stick_transparency", 0.9387820134336525, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 27")

cmd.set("stick_transparency", 0.9391084327674367, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 28")

cmd.set("stick_transparency", 0.9449599226812749, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 29")

cmd.set("stick_transparency", 0.9470536435209839, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 30")

cmd.set("stick_transparency", 0.95, f"N-acetyl_mureidomycin_A_PDB_kekulized.pdb_alphafold2 and state 31")

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
