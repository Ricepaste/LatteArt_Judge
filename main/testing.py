from RWM_testing import RandomWalkMatrix
# from ELO_testing import KendallTauCalculator
# import matplotlib.pyplot as plt
import subprocess

FLA = 0.0001
ADD_F = 0.00005
CON = 0.0001
for i in range(1, 3):
    FLASH = FLA + ADD_F * i
    RWM = RandomWalkMatrix(FLASH, CON)
    # ELO = KendallTauCalculator()
    print("FLASH: ", FLASH, "CON: ", CON)
    subprocess.run(["python", "main\\ELO_testing.py"], check=True)
    print("DONE")

