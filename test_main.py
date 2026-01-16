import main as gt

import numpy as np
import time

serv = gt.VolpeGreeterServicer()

serv.InitFromSeed(gt.pb.Seed(seed=1), None)
print("initialized")
serv.AdjustPopulationSize(gt.pb.PopulationSize(size=200), None)
print("adjusted size")
start = time.time()
last = time.time()

GENS=20

for i in range(GENS):
    serv.RunForGenerations(None, None)
    print(i, time.time() - start, time.time() - last)
    last = time.time()
print("ran for gens")
print((time.time() - start)/GENS)
res = serv.GetBestPopulation(gt.pb.PopulationSize(size=10), None)
serv.InitFromSeedPopulation(res, None)
res = serv.GetBestPopulation(gt.pb.PopulationSize(size=10), None)

for mem in res.members:
    print(mem.fitness)

result = serv.GetResults(gt.pb.PopulationSize(size=10), None)
for mem in result.members:
    print(mem.representation, mem.fitness)
