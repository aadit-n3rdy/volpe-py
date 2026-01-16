from typing import Any, override
import volpe_container_pb2 as pb
import common_pb2 as pbc
import volpe_container_pb2_grpc as vp
import grpc
import concurrent.futures
import threading

import tsplib95 as tsplib

problem = tsplib.load_problem('dsj1000.tsp')

NDIM=1000

MUTATION_RATE = 0.4
SWAP_PROB = 0.2
CXPROB = 0.9
REVERSAL_PROB = 2/500

DIVERSITY_COEFF = 1
DIVERSITY_MULTIP = 1-1e-2

FITNESS_DIV_COEFF = 0

NOISE_VAR = 0
NOISE_INC_MULTIP = 3.0
NOISE_DEC_MULTIP = 0.8

BASE_POPULATION_SIZE = 3000

import numpy as np

def inv_distance(s1, s2):
    # Create a DP table with dimensions (len(s1)+1) x (len(s2)+1)
    # using numpy for efficient array operations.
    s1 = s1[0]
    s2 = s2[0]
    return NDIM-np.sum(s1 == s2)

def fitness(x):
    return problem.trace_tours([x])[0]

def mutate(x):
    n_swaps = np.random.binomial(NDIM, SWAP_PROB)
    x = x[0].copy()
    for _ in range(n_swaps):
        i1 = np.random.randint(NDIM)
        i2 = np.random.randint(NDIM)
        buf = x[i1]
        x[i1] = x[i2]
        x[i2] = buf
    n_reversals = np.random.poisson(lam=REVERSAL_PROB*5934)
    for _ in range(n_reversals):
        i1 = np.random.randint(NDIM)
        i2 = np.random.randint(NDIM)
        if i1 > i2:
            i1, i2 = i2, i1
        x[i1:i2] = x[i1:i2][::-1]
    return (x, fitness(x))
def varAnd(popln):
    ogLen = len(popln)
    popln = select(popln, ogLen)
    newpopln = []
    for i in range(0, ogLen, 2):
        if np.random.random() < CXPROB:
            i1 = i
            i2 = i+1
            n1, n2 = crossover(popln[i1], popln[i2])
            newpopln.append(n1)
            newpopln.append(n2)
        else:
            newpopln.append(popln[i])
            newpopln.append(popln[i+1])
    return mutate_popln(newpopln)

def cross_raw(p0: np.ndarray, p1: np.ndarray):
    i1 = np.random.randint(NDIM)
    i2 = np.random.randint(NDIM)
    if i1 > i2:
        i2,i1=i1,i2
    pc = np.zeros(NDIM, np.int32)
    pc[i1:i2] = p0[i1:i2]
    doneSet = { x for x in p0[i1:i2] }
    ci = 0
    pi = 0
    while ci < len(pc):
        if ci == i1:
            ci = i2
            if ci >= len(pc):
                break
        while p1[pi] in doneSet:
            pi += 1
        pc[ci] = p1[pi]
        doneSet.add(p1[pi])
        pi += 1
        ci += 1
    return pc

def crossover(x: np.ndarray, y: np.ndarray):
    ind1 = cross_raw(x[0], y[0])
    ind2 = cross_raw(y[0], x[0])
    return (ind1, fitness(ind1)), (ind2, fitness(ind2))

def select(popln: list[tuple[np.ndarray, float]], newPop: int):
    newpopln = []
    # lcs_table = np.zeros(shape=(len(popln), len(popln)))
    # for i in range(BASE_POPULATION_SIZE-1):
    #     for j in range(i, BASE_POPULATION_SIZE):
    #         lcs_table[i][j] = inv_distance(popln[i], popln[j])
    # for i in range(BASE_POPULATION_SIZE-1):
    #     for j in range(i, BASE_POPULATION_SIZE):
    #         lcs_table[j][i] = lcs_table[i][j]
    global NOISE_VAR
    global DIVERSITY_COEFF
    global FITNESS_DIV_COEFF

    mean_fitness = np.mean([x[1] for  x in popln])

    popln_div = [(popln[i][0], popln[i][1], popln[i][1] + np.random.standard_normal()*np.sqrt(NOISE_VAR) - (popln[i][1] - mean_fitness)**2 * FITNESS_DIV_COEFF) for i in range(len(popln))]
    while len(newpopln) < newPop:
        choices = [popln_div[choice(popln)] for _ in range(3)]
        selected = min(choices, key=lambda x: x[2])
        newpopln.append((selected[0].copy(), selected[1]))
    return newpopln

def choice(popln: list[Any]):
    l = len(popln)
    idx = np.random.randint(0, l)
    return idx

def gen_ind():
    ind = np.random.permutation(NDIM)+1
    return (ind, fitness(ind))

def expand(popln: list[tuple[np.ndarray, float]], newPop: int):
    if len(popln) == 0:
        return [ gen_ind() for _ in range(newPop) ]
    if len(popln) >= newPop:
        return popln
    while len(popln) < newPop:
        x1 = popln[choice(popln)]
        x2 = popln[choice(popln)]
        y1, y2 = crossover(x1, x2)
        popln.append(y1)
        popln.append(y2)
    return popln


def get_random_list(popln: list[tuple[np.ndarray, float]], n: int):
    return [ popln[np.random.randint(len(popln))] for _ in range(n) ]

def mutate_popln(popln: list[tuple[np.ndarray, float]]):
    for i in range(len(popln)):
        if np.random.random() < MUTATION_RATE:
            popln[i] = mutate(popln[i])
    return popln

def popListTostring(popln: list[tuple[np.ndarray, float]]):
    indList : list[pb.ResultIndividual] = []
    for mem in popln:
        indList.append(
                pb.ResultIndividual(representation=np.array_str(mem[0]), 
                                    fitness=mem[1])
                )
    return pb.ResultPopulation(members=indList)

def bstringToPopln(popln: pbc.Population):
    popList = []
    for memb in popln.members:
        popList.append((np.frombuffer(memb.genotype, dtype=np.float32), memb.fitness))
    return popList

def adjustSize(popln: list[tuple[np.ndarray, float]], targetSize: int):
    print("ADJUSTSIZE called unexpectedly")
    if len(popln) < targetSize:
        return expand(popln, targetSize)
    else:
        return select(popln, targetSize)

def popListToBytes(popln: list[tuple[np.ndarray, float]]):
    indList : list[pbc.Individual] = []
    for mem in popln:
        indList.append(pbc.Individual(genotype=mem[0].astype(np.float32).tobytes(), fitness=mem[1]))
    return pbc.Population(members=indList)

class VolpeGreeterServicer(vp.VolpeContainerServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popln : list[tuple[np.ndarray, float]] = [ gen_ind() for _ in range(BASE_POPULATION_SIZE)  ]
        self.poplock = threading.Lock()
        self.last_best = np.inf

    @override
    def SayHello(self, request: pb.HelloRequest, context: grpc.ServicerContext):
        return pb.HelloReply(message="hello " + request.name)
    @override
    def InitFromSeed(self, request: pb.Seed, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            self.popln = [ gen_ind() for _ in range(BASE_POPULATION_SIZE)  ]
            return pb.Reply(success=True)
    @override
    def InitFromSeedPopulation(self, request: pbc.Population, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            ogLen = len(self.popln)
            seedPop = bstringToPopln(request)
            if self.popln == None:
                self.popln = []
            for p in seedPop:
                idx = np.random.randint(ogLen)
                self.popln[idx] = p
            # self.popln.extend(seedPop)

            # self.popln = select(self.popln, ogLen)

            return pb.Reply(success=True)
    @override
    def GetBestPopulation(self, request: pb.PopulationSize, context):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            if self.popln is None:
                return pbc.Population(members=[])
            popSorted = sorted(self.popln, key=lambda x: x[1])
            return popListToBytes(popSorted[:request.size])
    @override
    def GetResults(self, request: pb.PopulationSize, context):
        with self.poplock:
            if self.popln is None:
                return pbc.Population(members=[])
            popSorted = sorted(self.popln, key=lambda x: x[1])
            return popListTostring(popSorted[:request.size])
    @override
    def GetRandom(self, request: pb.PopulationSize, context):
        with self.poplock:
            if self.popln is None:
                return pbc.Population(members=[])
            popList = get_random_list(self.popln, request.size)
            return popListToBytes(popList)
    @override
    def AdjustPopulationSize(self, request: pb.PopulationSize, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        # context.abort(grpc.StatusCode.CANCELLED, "AdjustPopulationSize not allowed")
        # targetSize = request.size
        # # TODO: adjust to targetSize
        # with self.poplock:
        #     self.popln = adjustSize(self.popln, BASE_POPULATION_SIZE)
        #     return pb.Reply(success=True)
    @override
    def RunForGenerations(self, request: pb.PopulationSize, context):
        """Missing associated documentation comment in .proto file."""
        global DIVERSITY_COEFF
        global NOISE_VAR
        global NOISE_DEC_MULTIP
        global NOISE_INC_MULTIP
        with self.poplock:
            self.popln = varAnd(self.popln)
            # ogLen = len(self.popln)
            # popln = select(self.popln, ogLen)
            # newpopln = [ ]
            # for i in range(0, ogLen, 2):
            #     if np.random.random() < CXPROB:
            #         i1 = i
            #         i2 = i+1
            #         n1, n2 = crossover(popln[i1], popln[i2])
            #         newpopln.append(n1)
            #         newpopln.append(n2)
            #     else:
            #         n1, n2 = popln[i], popln[i+1]
            #         newpopln.append(n1)
            #         newpopln.append(n2)
            # self.popln = mutate_popln(newpopln)
            # self.popln = expand(self.popln, ogLen*2)
            # self.popln = select(self.popln, ogLen)
            newBest = min(self.popln, key=lambda x: x[1])[1]
            if newBest < self.last_best:
                self.last_best = newBest
                DIVERSITY_COEFF *= DIVERSITY_MULTIP
                NOISE_VAR *= NOISE_DEC_MULTIP
            else:
                DIVERSITY_COEFF /= DIVERSITY_MULTIP
                NOISE_VAR *= NOISE_INC_MULTIP
        return pb.Reply(success=True)

if __name__=='__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vp.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    server.add_insecure_port("0.0.0.0:8081")
    server.start()
    server.wait_for_termination()
