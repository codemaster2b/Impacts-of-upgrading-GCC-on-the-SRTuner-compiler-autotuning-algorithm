import os, subprocess, re
import os.path
import numpy as np
import random
import pandas as pd
import numpy as np
import gc
import time
import fast_histogram
import sys
import math
import json

from datetime import datetime

from anytree import Node, RenderTree, AsciiStyle, LevelOrderGroupIter, PostOrderIter
from anytree.dotexport import RenderTreeGraph
        
from configuration import Configuration
from flag import FlagInfo, FlagReader, FlagSet

# Define constant
FLOAT_MIN = (-1)*float('inf')
FLOAT_MAX = float('inf')
OUTFOLDER_NAME = "output"
OUTFILE_NAME = "output.txt"
DATAFILE_NAME = "data.csv"

def outToFileAndScreen(message):
    outToScreen(message)
    os.makedirs(OUTFOLDER_NAME, exist_ok=True)
    with open(OUTFOLDER_NAME + "/" + OUTFILE_NAME, "a") as ofp:
        ofp.write(message + "\n")

def outToScreen(message):
    print(message)

class Evaluator:
    def __init__(self, path, num_repeats):
        self.path = path
        self.num_repeats = num_repeats
    
    def build(self):
        assert 0, "Undefined"

    def run(self):
        assert 0, "Undefined"

    def evaluate(self):
        assert 0, "Undefined"

    def clean(self):
        assert 0, "Undefined"


class Tuner:
    def __init__(self, evaluator, name = "Base Tuner"):
        self.evaluator = evaluator
        self.name = name
        self.default_perf = sum([config.get_runtime(evaluator.benchmark)[x] for x in evaluator.benchmarkIndexes])
        self.visited = set()
    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf, new_best_perf = None, None, None
        i = 0
        while i < budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
        
            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if best_perf is None or perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            if i % 5 == 0:
                 outToFileAndScreen(f"{datetime.now()} {self.name} {config.gccVersion} {self.evaluator.benchmark} {i} / {budget}: {perf:.6f}s, best: {best_perf:.6f}s ({self.default_perf/best_perf:.3f}x)")

            if new_best_perf is None or best_perf < new_best_perf:
                new_best_perf = best_perf
            self.reflect_feedback(perfs)
        return best_opt_setting, best_perf

class RandomTuner(Tuner):
    def __init__(self, evaluator):
        super().__init__(evaluator, "Random Tuner")
        
    def generate_candidates(self, batch_size=1):
        random.seed(time.time())
        candidates = []
        for _ in range(batch_size):
            while True:
                # Avoid duplication
                self.evaluator.flagSet.randomize()
                opt_setting = self.evaluator.flagSet.zipEncodings(self.evaluator.flagSet.getEncodings())
                if opt_setting not in self.visited:
                    self.visited.add(opt_setting)
                    candidates.append(opt_setting)
                    break
                
        return candidates
    
    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting) for opt_setting in candidates]

    def reflect_feedback(self, perfs):
        # Random search. Do nothing
        pass

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
        
def convert_dict_to_encoding(self):
    pass

def delete_tree(root):
    for node in PostOrderIter(root):
        del node
    gc.collect()


def getUCT(N, n, r):
    c = np.sqrt(2)
    p_exploit = r/n
    p_explore = c*(np.sqrt(np.log(N)/n))
    return p_exploit + p_explore

def post_order_traversal(node, statTable = None):
    children = node.children
    if len(children) == 0: # leaf node
        return node.history
    else:
        aggr = []
        for child in node.children:
            data = post_order_traversal(child, statTable)
            if statTable != None:
                config = int(get_subencoding(child.encoding, -1))
                if node.depth >= len(statTable):
                    print(node)
                assert config < len(statTable[node.depth])
                statTable[node.depth][config].extend(data)
            aggr.extend(data)
        return aggr

def getNormHist(bins, data):
    count = fast_histogram.histogram1d(data, len(bins), range=(min(bins), max(bins)))
    tot = sum(count)
    num = len(data)
    assert(num >= tot)
    count = np.append(count, [num-tot])
    return count/num

def checksum(statTable, numAvailFlagKinds):
    chk = -1
    for i in range(numAvailFlagKinds):
        s = 0
        for j in range(len(statTable[i])):
            s += len(statTable[i][j])

        if chk == -1: 
            chk = s
        else:
            assert(chk == s)

def smooth(p):
    eps = 0.00000001
    numZeros = 0
    for v in p:
        if v == 0.0:
            numZeros += 1
    numNZ = len(p) - numZeros
    new_p = []
    for v in p:
        if v == 0.0:
            new_p.append(eps/numZeros)
        else:
            new_p.append(v-eps/numNZ)
    return new_p

# Kullback–Leibler divergence
def getKLD(p, q):
    # smoothing
    sp = smooth(p)
    sq = smooth(q)
    kl = 0
    for i in range(len(sp)):
        kl += sp[i]*np.log(sp[i]/sq[i])

    if(kl < 0):
        print(kl)
    assert(kl>=0)
    return kl

def shuffle_encoding(encoding, shuffle_mask):
    toks = encoding.split(',')
    new_encoding = ""
    for idx in shuffle_mask:
        if len(new_encoding):
            new_encoding += "," + toks[idx]
        else:
            new_encoding += toks[idx]
    return new_encoding

def get_subencoding(encoding, l):
    if l == -1:
        return encoding.split(',')[-1]
    else:
        toks = encoding.split(',')[0:l+1]
        return ",".join(toks)


# SRTuner as a standalone tuner
class SRTuner(Tuner):
    def __init__(self, evaluator):
        super().__init__(evaluator, "SRTuner")
        self.mod = SRTunerModule(evaluator, self.default_perf)

    def generate_candidates(self, batch_size=1):
        return self.mod.generate_candidates(batch_size)

    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting) for opt_setting in candidates]

    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)

class SRTunerModule():
    def __init__(self, evaluator, default_perf):
        self.default_perf = default_perf
        self.evaluator = evaluator

        # Create root node for multi-stage structure
        if self.default_perf is None or self.default_perf == FLOAT_MAX:
            self.root = Node(self.evaluator.flagSet.searchSpaceMapping[0], encoding="", num=0, reward=0, isDone=False, history=[])
        else:
            self.root = Node(self.evaluator.flagSet.searchSpaceMapping[0], encoding="", num=0, reward=0, isDone=False, history=[self.default_perf])

        self.best_perf = None
        self.worst_perf = None
        self.visited = set()
        self.trials = []
        self.shuffle_mask = []
        self.current_candidate_nodes = []
        self.batch_size = 1
        random.seed(time.time())

    # This will give you candidate and leaf node
    def reward_func(self, perf, best_perf, num_trials, min_trials=10):
        # hyperparameters for reward calc
        # [TODO] Design reward policy that has less hyperparams
        C = 60
        max_reward=100
        reward_margin = -0.05
        window = max(min_trials, num_trials-500)

        reward = 0
        ratio = best_perf/perf
        if num_trials > window and ratio>1+reward_margin:
            reward = min(C*ratio, max_reward)
        return max(reward, 0)

    def traverse(self, enable_expansion):
        cur_node = self.root
        while True:
            assert not self.root.isDone, "Search space is completely explored."
            numChildren = len(cur_node.children)
            numConfigs = len(self.evaluator.flagReader.searchSpace[cur_node.name].configs)

            if numChildren < numConfigs:
                # Without node expansion
                new_encoding = str(cur_node.encoding)
                if not enable_expansion:
                    # Pick random options w/o real expansion
                    for i in range(cur_node.depth, len(self.evaluator.flagReader.searchSpace)):
                        opt_name = self.evaluator.flagSet.searchSpaceMapping[i]
                        numConfigs = len(self.evaluator.flagReader.searchSpace[opt_name].configs)
                        chosenConfig = random.randint(0,numConfigs-1)
                        if len(new_encoding):
                            new_encoding += "," + str(chosenConfig)
                        else:
                            new_encoding = str(chosenConfig)
                    return new_encoding

                assert(enable_expansion)
                # With node expansion
                # Need to explore. Random sampling
                while True:
                    chosenConfig = random.randint(0,numConfigs-1)
                    if not any([get_subencoding(child.encoding, -1) == str(chosenConfig) for child in cur_node.children]):
                        break

                # Attach new config
                if len(cur_node.encoding):
                    new_encoding = cur_node.encoding + "," + str(chosenConfig)
                else:
                    new_encoding = str(chosenConfig)

                if cur_node.depth+1 == len(self.evaluator.flagReader.searchSpace):
                     # leaf node
                    return Node("leaf", parent=cur_node, encoding=new_encoding, num=0, reward=0, isDone=True, history=[])
                else:
                    # other nodes
                    cur_node = Node(self.evaluator.flagSet.searchSpaceMapping[cur_node.depth+1], parent=cur_node,
                                encoding=new_encoding, num=0, reward=0, isDone=False, history=[])
            else:
                # Balance
                UCT_N = cur_node.num
                max_id, max_uct = -1, -1
                zero_nodes = []
                for i in range(numChildren):
                    childNode = cur_node.children[i]
                    if childNode.isDone == False:
                        if childNode.num == 0:
                            zero_nodes.append(childNode)
                        else:
                            uct = getUCT(cur_node.num, childNode.num, childNode.reward)
                            if uct > max_uct:
                                max_id, max_uct = i, uct

                if max_id == -1:
                    if len(zero_nodes):
                        cur_node = cur_node.children[random.randint(0, len(zero_nodes)-1)]
                    else:
                        # All done
                        cur_node.isDone = True
                        cur_node = cur_node.parent
                else:
                    cur_node = cur_node.children[max_id]
        assert 0, "Should not reach here"


    # Expands multistatage structure with provided encodings. 
    # Model-based tuning methods will use this to reflect runs w/ actual hardware. 
    def expand(self, opt_settings):
        for opt_setting in opt_settings:
            depth = 0
            cur_node = self.root
        
            while depth < len(self.evaluator.flagReader.searchSpace):
                chosenConfig = opt_setting[cur_node.name]
                found_child = None
                for child in cur_node.children:
                    if get_subencoding(child.encoding, -1) == str(chosenConfig):
                        found_child = child
                        break

                if len(cur_node.encoding):
                    new_encoding = cur_node.encoding + "," + str(chosenConfig)
                else:
                    new_encoding = str(chosenConfig)

                if found_child is None:
                    if depth+1 == len(self.evaluator.flagReader.searchSpace):
                        new_node = Node("leaf", encoding=new_encoding, num=0, reward=0, 
                                isDone=True, history=[], parent=cur_node)
                    else:
                        new_node = Node(self.evaluator.flagSet.searchSpaceMapping[depth+1], encoding=new_encoding, num=0, reward=0, 
                                isDone=False, history=[], parent=cur_node)
                    cur_node = new_node
                    
                else:
                    cur_node = found_child
                    assert(found_child.name != "leaf")
                    
                depth = cur_node.depth
            
            assert(cur_node.name == "leaf")
            self.current_candidate_nodes.append(cur_node)
            self.visited.add(cur_node.encoding)

        return None
                
    # generate
    def generate_candidates(self, batch_size = 1, enable_expansion=True):
        self.batch_size = batch_size
        candidates = []
        if enable_expansion:
            # This mode expands nodes during traversal. 
            # It returns a list of leaf nodes in the multistage structure.
            self.current_candidate_nodes = [] # This remembers leaf nodes for each encoding
            for _ in range(batch_size):
                leaf_node = self.traverse(enable_expansion=True)
                assert(leaf_node.encoding not in self.visited)
                zippedOpt = self.evaluator.flagSet.zipMappedString(leaf_node.encoding)
                candidates.append(zippedOpt)
                self.current_candidate_nodes.append(leaf_node)
        else:
            # This mode *DOES NOT* expand nodes during the traversal. 
            # Model-based tuning methods will use this mode to find the promising candidates with their cost model. 
            # Since cost model may not be accurate enough, SRTuner does not reflect runs w/ cost model. 
            # It returns a list of optimization settings.
            for _ in range(batch_size):
                encoding = self.traverse(enable_expansion=False)
                zippedOpt = self.evaluator.flagSet.zipMappedString(encoding)
                candidates.append(zippedOpt)
        return candidates

    def remap(self, numBins = 100, minSamples=15):
        # Collect stats
        bins = np.linspace(self.best_perf.sum(), self.worst_perf.sum(), numBins)
        statTable = []
        for i, opt_name in enumerate(self.evaluator.flagSet.searchSpaceMapping):
            num_configs = len(self.evaluator.flagReader.searchSpace[opt_name].configs)
            statTable.append([[] for i in range(num_configs)])

        post_order_traversal(self.root, statTable)

        # Verify
        checksum(statTable, len(self.evaluator.flagReader.searchSpace))

        # Distribution-based impact estimation w/ KL divergence
        t_hist, t_kls = 0, 0
        klData = []
        for optId in range(len(self.evaluator.flagReader.searchSpace)):
            perfs = statTable[optId]
            num_configs = len(perfs)
            flag = self.evaluator.flagSet.searchSpaceMapping[optId]
            if num_configs == 1:
                kl = 0
            else:
                if num_configs > 2:
                    min_avg = FLOAT_MAX
                    max_avg = FLOAT_MIN
                    min_idx = -1
                    max_idx = -1

                    for j in range(num_configs):
                        if len(perfs[j]) == 0:
                            continue

                        avg = np.mean([perf for perf in perfs[j] if perf != FLOAT_MAX])

                        if avg > max_avg:
                            max_avg, max_idx = avg, j

                        if avg < min_avg:
                            min_avg, min_idx = avg, j

                    if min_idx > max_idx:
                        tmp = min_idx
                        min_idx = max_idx
                        max_idx = tmp

                    if( (min_idx < max_idx) and (min_idx>=0) ):
                        p = perfs[min_idx]
                        q = perfs[max_idx]
                        labels = [str(min_idx), str(max_idx)]
                    else:
                        p = []
                        q = []

                else:
                    p = perfs[0]
                    q = perfs[1]
                    labels = ["0", "1"]

                # Impact
                if (len(p) >= minSamples) and (len(q) >= minSamples):
                    # without smoothing
                    dist0 = getNormHist(bins, p)
                    dist1 = getNormHist(bins, q)
                    # smooth dists internally
                    kl = getKLD(dist0, dist1)
                else:
                    kl = 0
                del perfs, p, q
            klData.append([kl, flag])
        del bins, statTable

        # Order optimization in the order of its impact
        dfKlData = pd.DataFrame(klData, columns=["KL", "optimization"])
        dfKlData.sort_values(by=["KL"], inplace=True, ascending=False)
        self.shuffle_mask = dfKlData["optimization"].index.values.tolist()

        # Update the impact
        self.evaluator.flagSet.searchSpaceMapping = dfKlData["optimization"].values.tolist()

        del klData, dfKlData
        gc.collect()

        self.visited = set()
        root_num = self.root.num
        delete_tree(self.root)
        # Create root node for multi-stage structure
        if self.default_perf is None or self.default_perf == FLOAT_MAX:
            self.root = Node(self.evaluator.flagSet.searchSpaceMapping[0], encoding="", num=0, reward=0, isDone=False, history=[])
        elif self.default_perf != FLOAT_MAX:
            self.root = Node(self.evaluator.flagSet.searchSpaceMapping[0], encoding="", num=0, reward=0, isDone=False, history=[self.default_perf])
        df_trials = pd.DataFrame(self.trials, columns=["encoding", "performance"])

        encodings = df_trials["encoding"].values
        perfs = df_trials["performance"].values
        numTrials = len(perfs)

        for i in range(numTrials):
            perf = perfs[i]
            encoding = encodings[i]
            encoding = shuffle_encoding(encoding, self.shuffle_mask)
            self.trials[i][0] = encoding

            depth = 0
            cur_node = self.root
            while depth < len(self.evaluator.flagReader.searchSpace):
                sub_encoding = get_subencoding(encoding, depth)
                found = False
                for child in cur_node.children:
                    if child.encoding == sub_encoding:
                        found = True
                        break

                if found:
                    cur_node = child
                else:
                    if depth+1 == len(self.evaluator.flagReader.searchSpace):
                        new_node = Node("leaf", encoding=sub_encoding, num=0, reward=0, isDone=True, history=[], parent=cur_node)
                    else:
                        new_node = Node(self.evaluator.flagSet.searchSpaceMapping[depth+1], encoding=sub_encoding, num=0, reward=0, isDone=False, history=[], parent=cur_node)
                        
                    cur_node = new_node
                depth = cur_node.depth
            self.backpropagate(cur_node, perf)

        assert(root_num == self.root.num)
        del perfs, encodings, df_trials
        gc.collect()

    # update
    def backpropagate(self, leaf_node, perf):
        assert leaf_node is not None
        assert perf is not None

        node_list = [ leaf_node ]
        self.visited.add(leaf_node.encoding)
        node_list.extend(reversed(leaf_node.ancestors))
        assert(len(node_list) == len(self.evaluator.flagReader.searchSpace)+1)  # num_opts + leaf node
        root = node_list[-1]
        assert(root.depth == 0)
        
        for node in node_list:
            best = FLOAT_MAX
            if self.best_perf is not None:
                best = self.best_perf
            reward = self.reward_func(perf, best, len(node.history))
            node.num += 1
            node.reward += reward
            node.history.append(perf)
            node.history.sort(reverse=True)

    def reflect_feedback(self, perfs, remap_freq = 100):
        for leaf_node, perf in zip(self.current_candidate_nodes, perfs):
            self.backpropagate(leaf_node, perf)
            if self.best_perf is None or perf < self.best_perf:
                self.best_perf = perf
            if self.worst_perf is None or (perf != FLOAT_MAX and perf > self.worst_perf):
                self.worst_perf = perf
            self.trials.append([leaf_node.encoding, perf])
            
        self.current_candidate_nodes = []
        if self.root.num % remap_freq == 0:
            self.remap()

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, None
        i = 0
        while i<budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            i += len(candidates)
            
            perfs = self.evaluate_candidates(candidates)
            self.reflect_feedback(perfs)
            for opt_setting, perf in zip(candidates, perfs):
                if best_perf is None or perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting

            print(f"[{i}] {best_perf:.6f}")
        
        return best_opt_setting, best_perf

    def extract_synergy(self):
        assert 0, "[TODO]"

def makeJtime():
    processHandles = []
    commands = f"""cd ../jtime/src;
    make;
    """
    subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True).wait()

def calibrateNewParam(multiEval, benchmark, i, startTime, otherSamples = []):
    samples = []
    samples.extend(otherSamples)
    for j in range(config.prec_samples - len(otherSamples)):
        perf = multiEval.calibrateParameter(i)
        samples.append(perf)
        if j % 5 == 0:
            outToFileAndScreen(f"{datetime.now()}: {config.gccVersion} {benchmark} {i} thread = {multiEval.num_threads}, loops = {multiEval.num_loops[i]}, finfos = {multiEval.num_finfos[i]}, sample {j+1} / {config.prec_samples} = {perf:.6f}")
    mean = np.mean(samples)
    prec = 300*np.std(samples)/mean
    config.set_runtime(benchmark, i, round(mean,6))
    config.set_prec(benchmark, i, round(prec,6))
    config.set_threads(benchmark, i, multiEval.num_threads)
    config.set_finfos(benchmark, i, multiEval.num_finfos[i])
    config.set_loops(benchmark, i, multiEval.num_loops[i])
    config.set_caltime(benchmark, i, round((datetime.now() - startTime).total_seconds(),0))
    outToFileAndScreen("- run time: " + str(config.get_runtime(benchmark)[i]))
    outToFileAndScreen("- percent consistency: " + str(config.get_prec(benchmark)[i]))
    outToFileAndScreen("- target consistency: " + str(config.precTarget))
    outToFileAndScreen("- time calibrating parameter: " + str(config.get_caltime(benchmark)[i]))
    return samples

def getNewLoops(loops, mean, prec):
    maxFactor = math.sqrt(config.precMaxRuntime / mean)
    factor = min(maxFactor, prec / config.precTarget)    
    newLoops = math.ceil(factor * factor * loops)
    return newLoops

def calibrateSome(path, flagReader, benchmark, datafiles):
    multiEval = cBenchEvaluator(path, benchmark, datafiles, flagReader, 0)
    multiEval.num_loops = [1 for x in multiEval.parameters]
    multiEval.num_finfos = [1 for x in multiEval.parameters]
    outToFileAndScreen(f"{datetime.now()}: Calibrating {benchmark}")
    for i in range(len(multiEval.parameters)):
        if len(config.get_runtime(benchmark)) <= i:
            config.set_runtime(benchmark, i, 0)
        if len(config.get_prec(benchmark)) <= i:
            config.set_prec(benchmark, i, 0)
        if len(config.get_threads(benchmark)) <= i:
            config.set_threads(benchmark, i, 0)
        if len(config.get_finfos(benchmark)) <= i:
            config.set_finfos(benchmark, i, 0)
        if len(config.get_loops(benchmark)) <= i:
            config.set_loops(benchmark, i, 0)
        if len(config.get_caltime(benchmark)) <= i:
            config.set_caltime(benchmark, i, 0)

        startTime = datetime.now()
        if i in datafiles and (config.get_threads(benchmark)[i] <= 0 or config.get_loops(benchmark)[i] <= 0 or config.get_finfos(benchmark)[i] <= 0):
            outToFileAndScreen(f"{datetime.now()}: Dataset {i}")
            threadNum = 2
            threadLoops = {}
            threadFinfos = {}
            threadRuntimes = {}
            threadPrecs = {}
            threadRatings = []
            threadNums = []
            searchMode = 0
            
            #temporary override
            #threadNum = 5
            #searchMode = 3
                    
            while searchMode < 4 and threadNum > 0:
                #initial thread work
                multiEval.num_threads = threadNum
                multiEval.calibrateParameterPre({})
                multiEval.num_finfos[i] = 2
                multiEval.num_loops[i] = 2
                config.prec_samples = 10
                calibrateNewParam(multiEval, benchmark, i, startTime)

                multiEval.num_finfos[i] = 0
                #temporary override read finfo from original file
                finfoFileName = f"{path}{i}/_ccc_info_datasets"
                if os.path.exists(finfoFileName):
                    f = open(finfoFileName, "r")
                    multiEval.num_finfos[i] = int(f.readlines()[8].strip())
                    print(f"finfo set by file to {multiEval.num_finfos[i]}.")
                else:
                    print(f"file {finfoFileName} does not exist.")
                
                #rapid determine finfos by minimum factor
                if multiEval.num_finfos[i] == 0:
                    multiEval.num_finfos[i] = 2
                    initial_runtime = config.get_runtime(benchmark)[i]
                    while config.get_runtime(benchmark)[i] < config.precMaxRuntime/1.5 and config.get_runtime(benchmark)[i] / initial_runtime < config.precFinfoMult and multiEval.num_finfos[i] < 1000000:
                        multiEval.num_finfos[i] = math.ceil(multiEval.num_finfos[i] * 2)
                        calibrateNewParam(multiEval, benchmark, i, startTime)

                    multiEval.num_finfos[i] /= 2
                    calibrateNewParam(multiEval, benchmark, i, startTime)
                    while config.get_runtime(benchmark)[i] < config.precMaxRuntime/1.5 and config.get_runtime(benchmark)[i] / initial_runtime < config.precFinfoMult and multiEval.num_finfos[i] < 1000000:
                        multiEval.num_finfos[i] = math.ceil(multiEval.num_finfos[i] * 1.2)
                        calibrateNewParam(multiEval, benchmark, i, startTime)
                    if multiEval.num_finfos[i] >= 1000000:
                        multiEval.num_finfos[i] = 1
                    else:
                        multiEval.num_finfos[i] = math.ceil(multiEval.num_finfos[i] / min(1.1, config.get_runtime(benchmark)[i] / initial_runtime / config.precFinfoMult))
                        
                #rapid determine loops by minimum factor
                initial_runtime = config.get_runtime(benchmark)[i]
                while config.get_runtime(benchmark)[i] / initial_runtime < 1.5 and multiEval.num_loops[i] < 1000000:
                    multiEval.num_loops[i] = math.ceil(multiEval.num_loops[i] * 2)
                    calibrateNewParam(multiEval, benchmark, i, startTime)
                    
                multiEval.num_loops[i] /= 2
                calibrateNewParam(multiEval, benchmark, i, startTime)
                while config.get_runtime(benchmark)[i] / initial_runtime < 1.5 and multiEval.num_loops[i] < 1000000:
                    multiEval.num_loops[i] = math.ceil(multiEval.num_loops[i] * 1.2)
                    calibrateNewParam(multiEval, benchmark, i, startTime)
                if multiEval.num_loops[i] >= 1000000:
                    multiEval.num_loops[i] = 1
                else:
                    multiEval.num_loops[i] = math.ceil(multiEval.num_loops[i] / min(1.1, config.get_runtime(benchmark)[i] / initial_runtime / 1.5))
                    
                min_loops = multiEval.num_loops[i]
                        
                #rough compute loops
                config.prec_samples = 100
                sampleLoops = multiEval.num_loops[i]
                samples = calibrateNewParam(multiEval, benchmark, i, startTime)
                newLoops = getNewLoops(multiEval.num_loops[i], config.get_runtime(benchmark)[i], config.get_prec(benchmark)[i])
                multiEval.num_loops[i] = max(newLoops, min_loops)
                if newLoops > min_loops:
                    sampleLoops = multiEval.num_loops[i]
                    samples = calibrateNewParam(multiEval, benchmark, i, startTime)
                    newLoops = getNewLoops(multiEval.num_loops[i], config.get_runtime(benchmark)[i], config.get_prec(benchmark)[i])
                    if newLoops < multiEval.num_loops[i]:
                        newLoops = math.ceil((newLoops + multiEval.num_loops[i])/2.0)
                    multiEval.num_loops[i] = max(newLoops, min_loops)
                if newLoops > min_loops:
                    sampleLoops = multiEval.num_loops[i]
                    samples = calibrateNewParam(multiEval, benchmark, i, startTime)
                    newLoops = getNewLoops(multiEval.num_loops[i], config.get_runtime(benchmark)[i], config.get_prec(benchmark)[i])
                    if newLoops < multiEval.num_loops[i]:
                        newLoops = math.ceil((newLoops + multiEval.num_loops[i])/2.0)
                    multiEval.num_loops[i] = max(newLoops, min_loops)

                #final compute loops
                config.prec_samples = 500
                if sampleLoops == multiEval.num_loops[i]:
                    calibrateNewParam(multiEval, benchmark, i, startTime, samples)
                else:
                    calibrateNewParam(multiEval, benchmark, i, startTime)
                        
                threadLoops[threadNum] = config.get_loops(benchmark)[i]
                threadFinfos[threadNum] = config.get_finfos(benchmark)[i]
                threadRuntimes[threadNum] = config.get_runtime(benchmark)[i]
                threadPrecs[threadNum] = config.get_prec(benchmark)[i]
                        
                #if 10% precision instead of 2%, thats a 25x multiplier, but the runtime itself could be +/- 10%                        
                newRating = config.get_runtime(benchmark)[i] * math.pow(config.get_prec(benchmark)[i] / config.precTarget, 2)
                minNewRating = newRating * (1 - config.get_prec(benchmark)[i]/200)
                maxNewRating = newRating * (1 + config.get_prec(benchmark)[i]/200)
                        
                threadRatings.append(maxNewRating)
                threadNums.append(threadNum)
                outToFileAndScreen(f"{datetime.now()}: {benchmark} {i} thread = {threadNum}, loops = {multiEval.num_loops[i]}, finfos = {multiEval.num_finfos[i]}, rating = {newRating:.6f}")
                if maxNewRating == min(threadRatings) and searchMode < 2:
                    threadNum = threadNum + 2
                    searchMode = 0
                elif maxNewRating == min(threadRatings):
                    searchMode = 4 #done
                elif minNewRating > min(threadRatings) and searchMode == 0:
                    searchMode = 1
                    threadNum = threadNum + 1
                elif minNewRating > min(threadRatings) and searchMode < 3:
                    searchMode = searchMode + 1
                    while threadNum in threadNums:
                        threadNum = threadNum - 1 # go backwards
                elif searchMode < 2:
                    threadNum = threadNum + 1
                elif searchMode == 2:
                    while threadNum in threadNums:
                        threadNum = threadNum - 1 # go backwards
                else:
                    searchMode = 4
                            
            #finalize choice
            bestIndex = threadRatings.index(min(threadRatings))
            threadNum = threadNums[bestIndex]
            config.set_runtime(benchmark, i, threadRuntimes[threadNum])
            config.set_prec(benchmark, i, threadPrecs[threadNum])
            config.set_threads(benchmark, i, threadNum)
            config.set_finfos(benchmark, i, threadFinfos[threadNum])
            config.set_loops(benchmark, i, threadLoops[threadNum])
            config.set_caltime(benchmark, i, round((datetime.now() - startTime).total_seconds(),0))
            config.write()
                
            configName = config.fileName.split(".ini")[0]
            datePart = str(datetime.now()).split(" ")[0]
            timePart = str(datetime.now()).split(" ")[1]
            if not os.path.exists(OUTFOLDER_NAME + "/calibrations.csv"):
                with open(OUTFOLDER_NAME + "/calibrations.csv", "a") as data:
                    data.write("DATE,TIME,ENV,GCC,BENCHMARK,PRECTARGET,FINFOMULT,MAXRT,DATAFILE,THREADS,FINFOS,LOOPS,RUNTIME,PREC,CALTIME\n")
            with open(OUTFOLDER_NAME + "/calibrations.csv", "a") as data:
                data.write(f"{datePart},{timePart},{configName},{config.gccVersion},{benchmark},{config.precTarget:.2f},{config.precFinfoMult:.2f},{config.precMaxRuntime:.2f},{i},{config.get_threads(benchmark)[i]},{config.get_finfos(benchmark)[i]},{config.get_loops(benchmark)[i]},{config.get_runtime(benchmark)[i]},{config.get_prec(benchmark)[i]},{config.get_caltime(benchmark)[i]}\n")
            
        elif i in datafiles and (config.get_runtime(benchmark)[i] <= 0 or config.get_prec(benchmark)[i] <= 0):
            multiEval.num_threads = config.get_threads(benchmark)[i]
            multiEval.calibrateParameterPre({})
            multiEval.num_finfos[i] = config.get_finfos(benchmark)[i]
            multiEval.num_loops[i] = config.get_loops(benchmark)[i]
            config.prec_samples = 500
            calibrateNewParam(multiEval, benchmark, i, startTime)
            config.write()
        config.write()

# Define tuning task
class cBenchEvaluator(Evaluator):
    def __init__(self, path, benchmark, benchmarkIndexes, flagReader, line):
        super().__init__(path, 1)
        self.artifact = "a.out"
        self.benchmark = benchmark
        self.benchmarkIndexes = benchmarkIndexes
        self.parameters = []
        self.num_threads = 1
        self.num_loops = []
        self.num_finfos = []
        self.readParameters()
        self.flags = None
        self.flagReader = flagReader
        self.flagSet = FlagSet(flagReader)
        self.flagSet.useImplies = True
        self.flagSet.useBenefits = False
        self.isRunAblative = "N"
        self.aFlag = ""
        self.runLine = line

    def build(self):
        start = datetime.now()
        processHandles = []
        for i in range(self.num_threads):
            if not os.path.isdir(f"{self.path}{i}"):
                commands = f"cd {self.path}/..; rm -rf src{i}; cp -r src src{i};"
                processHandles.append(subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True))
            else:
                commands = f"cd {self.path}{i}; rm -f {self.artifact} *.o *.a *.s *.i *.I"
                processHandles.append(subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True))
        for handle in processHandles:
            handle.wait()
            

        self.flags = self.flagSet.getCompilerOptions()
        commands = f"cd {self.path}0; gcc-{config.gccVersion} -c -I./ *.c {self.flags} 2>/dev/null; gcc-{config.gccVersion} -o {self.artifact} -fopenmp -lm *.o -lm"
        #commands = f"""cd {self.path}0; make clean > /dev/null 2>/dev/null; make -j8 CCC_OPTS_ADD="{self.flags}" LD_OPTS=" -o {self.artifact} -fopenmp" > /dev/null 2>/dev/null;"""
        #outToFileAndScreen(f"Build commands {commands}")
        subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True).wait()
        
        for i in range(1, self.num_threads):
            commands = f"cp {self.path}0/{self.artifact} {self.path}{i}"
            #outToFileAndScreen(f"Build commands {commands}")            
            processHandles.append(subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True))
        for handle in processHandles:
            handle.wait()

        #outToFileAndScreen(f"Build time {(datetime.now()-start).total_seconds():.6f}s")

        # Check if build fails
        for i in range(self.num_threads):
            if not os.path.exists(f"{self.path}{i}/{self.artifact}"):
                outToFileAndScreen(f"Build failed")
                return -1
        return 0

    def run(self, index):
        loops = 1
        if len(self.num_loops) > index:
            loops = self.num_loops[index]
        finfos = 1
        if len(self.num_finfos) > index:
            finfos = self.num_finfos[index]
        parameter = ""
        if len(self.parameters) > index:
            parameter = self.parameters[index]
    
        runs = []
        while(os.path.isfile("./pause.python.txt")):
            time.sleep(1)

        # Repeat the measurement and get the averaged execution time
        iterationRuns = []
        processHandles = []
        for i in range(self.num_threads):
            commands = f"cd {self.path}{i}; echo {finfos} > _finfo_dataset; ../../../../jtime/src/jtime 1 {loops} ./a.out {parameter} > out.o 2>&1"
            #commands = f"""cd {self.path}{i}; ./_ccc_check_output.clean; ./__run {index} 2>&1"""
            #outToFileAndScreen(commands)
            processHandles.append(subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True))
        for handle in processHandles:
            handle.wait()

        expectedSamples = 1

        #still missing loops
        outputFileSizes = []
        for i in range(self.num_threads):
            secs = 0.0
            samples = 0
            
            f = open(f"{self.path}{i}/out.o", "rb")
            raw = f.read()
            #raw = processHandles[i].stdout.read()
            outputFileSizes.append(len(raw))
            
            for out in raw.split(b'\n'):
                if out.startswith(b'real'):
                    line = out.decode("ascii");
                    line = line.replace("real\t", "")
                    nums = re.findall("\d*\.?\d+", line)
                    if len(nums) == 2:
                    	secs += float(nums[0])*60+float(nums[1])
                    	samples += 1
                    else:
                    	outToFileAndScreen(f"Error in time line [{line}]")
            iterationRuns.append(secs)
            
            if samples != expectedSamples:
                outToFileAndScreen(f"Samples error for {self.path} expected {expectedSamples} got {samples}")
            
        runs = [x for x in iterationRuns]

        # save all results
        configName = config.fileName.split(".ini")[0]
        datePart = str(datetime.now()).split(" ")[0]
        timePart = str(datetime.now()).split(" ")[1]
        compressedFlags = self.flagSet.zipEncodings(self.flagSet.getEncodings())
        os.makedirs(OUTFOLDER_NAME, exist_ok=True)
        if not os.path.exists(OUTFOLDER_NAME + "/" + DATAFILE_NAME):
            with open(OUTFOLDER_NAME + "/" + DATAFILE_NAME, "a") as data:
                data.write("DATE,TIME,ENV,GCC,BENCHMARK,DATAFILE,TUNER,FLAGS,NUM_FINFOS,NUM_LOOPS,NUM_THREADS,PERF,OUTFILE,LINE,ABLATIVE,FLAG\n")
        with open(OUTFOLDER_NAME + "/" + DATAFILE_NAME, "a") as data:
            for i in range(len(runs)):
                data.write(f"{datePart},{timePart},{configName},{config.gccVersion},{self.benchmark},{index},{config.tuner},{compressedFlags},{finfos},{loops},{self.num_threads},{runs[i]},{outputFileSizes[i]},{self.runLine},{self.isRunAblative},{self.aFlag}\n")

        return np.mean(runs)

    def evaluate(self, zippedOpt):
        self.isRunAblative = "N"
        self.aFlag = ""
        return self.runAll(zippedOpt)
        
    def runAll(self, zippedOpt):
        self.flagSet.setZipped(zippedOpt)
        self.flags = self.flagSet.getCompilerOptions()
        
        self.num_threads = max(config.get_threads(self.benchmark))
        error = self.build()
        if error == -1 or len(self.num_loops) != len(self.parameters):
            outToFileAndScreen(f"Build error 2 for {self.path}")
            outToFileAndScreen(f"{len(self.num_loops)} {len(self.parameters)}")
            return FLOAT_MAX

        means = []
        for i in range(len(self.parameters)):
            if i in self.benchmarkIndexes:
                self.num_threads = config.get_threads(self.benchmark)[i]
                means.append(self.run(i))
            else:
                means.append(0)

        # adjust for different datafile run times
        if config.doCombineFairly:
            adj_means = []
            runtimes = 0
            for i in self.benchmarkIndexes:
                adj_means.append(means[i] / config.get_runtime(self.benchmark)[i])
                runtimes += config.get_runtime(self.benchmark)[i]
            return np.mean(adj_means) * runtimes
        else:
            return sum(means)

    def calibrateParameterPre(self, chosen):
        self.flagSet.setChosen(chosen)
        self.flags = self.flagSet.getCompilerOptions()
        #outToFileAndScreen(f"Build flags {flags}")
        error = self.build()
        if error == -1:
            outToFileAndScreen(f"Build error 1 for {self.path}")

    def calibrateParameter(self, index):
        mean = self.run(index)
        return mean

    def readParameters(self):
        self.parameters = []
        fileName = self.path + "/_parameters"
        if os.path.isfile(fileName):
            file = open(fileName)
            for dataset in file.read().split("\n"):
                if len(dataset) > 0:
                    self.parameters.append(dataset)
                    for paramFileName in dataset.split(" "):
                        if os.path.isfile(self.path + "/" + paramFileName):
                            f = open(f"{self.path}/{paramFileName}", "rb")
                            raw = f.read()
                            f.close()
            file.close()

# GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME
def processQueue(config, flagReader):
    file=open(config.queue)
    lines = file.read().split("\n")
    
    # build multiplex file
    #with open(config.queue + ".mp", "w") as data:
    #    data.write("GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME\n")
    #    datafileResults = {}
    #    for i in range(1, len(lines)):
    #        items = lines[i].split(',')
    #        if len(items) >= 19:
    #            key = f"{items[2]}{items[3]}"
    #            if key not in datafileResults:
    #                datafileResults[key] = []
    #            datafileResults[key].append(items)
    #    for key in datafileResults:
    #        for i in range(len(datafileResults[key])):
    #            for j in range(len(datafileResults[key])):
    #                if i != j:
    #                    items = datafileResults[key][i].copy()
    #                    items[14] = datafileResults[key][j][14]
    #                    items[15] = ""
    #                    items[16] = ""
    #                    items[18] = ""
    #                    data.write(",".join(items) + "\n")
    
    # run operation
    for i in range(1, len(lines)):
        items = lines[i].split(',')
        if len(items) >= 19:
            gccVersion = items[0]
            if gccVersion in config.supportedGccVersions:
                config.gccVersion = gccVersion
            tunerName = items[1]
            benchmark = items[2]
            datafiles = json.loads(items[3].replace(':',','))
            iterations = int(items[4])
            config.precFinfoMult = float(items[5])
            config.precTarget = float(items[6])
            config.precMaxRuntime = float(items[7])
            config.doAblation = True if items[8] == "True" else False
            config.doBenefits = True if items[9] == "True" else False
            config.doImplies = True if items[10] == "True" else False
            
            tuner = None
            limitReached = False
            if len(items[11]) == 0 or len(items[15]) < 3:
                path = f"../benchmarks/cBench/{benchmark}/src"
                calibrateSome(path, flagReader, benchmark, datafiles)
                multiEval = cBenchEvaluator(path, benchmark, datafiles, flagReader, i)
                multiEval.num_loops = config.get_loops(benchmark)
                multiEval.num_finfos = config.get_finfos(benchmark)
                if tunerName == "random":
                    tuner = RandomTuner(multiEval)
                elif tunerName == "srtuner":
                    tuner = SRTuner(multiEval)

                #temporary override
                #while not limitReached and config.precFinfoMult > 1.1:
                #    config.precFinfoMult = config.precFinfoMult / 2
                #    for df in datafiles:
                #        if len(config.get_runtime(benchmark)) > df and config.get_runtime(benchmark)[df] > config.precMaxRuntime/1.5:
                #            limitReached = True
                #config.precFinfoMult = float(items[5])

            # run operation
            if len(items[11]) == 0 and not limitReached:
                if tuner is not None:
                    uid = config.queue[14:-4]
                    startTime = datetime.now()
                    OUTFOLDER_NAME = f"output/{config.gccVersion}"
                    OUTFILE_NAME = f"{uid}.{tuner.name}.{benchmark}.output.txt"
                    DATAFILE_NAME = f"{uid}.{tuner.name}.{benchmark}.data.csv"
                
                    config.tuner = tuner.name
                    outToFileAndScreen(f"Tuning {benchmark} w/ {tuner.name}")
                    best_opt_setting, best_perf = tuner.tune(iterations)
                    if best_opt_setting is not None:
                        outToFileAndScreen(f"Tuning {benchmark} w/ {tuner.name}: {tuner.default_perf:.6f}/{best_perf:.6f} = {tuner.default_perf/best_perf:.6f}x")
                        items[11] = f"{tuner.default_perf:.6f}"
                        items[12] = f"{best_perf:.6f}"
                        items[13] = f"{tuner.default_perf/best_perf:.6f}"
                        items[14] = best_opt_setting
                        items[16] = f"{round((datetime.now() - startTime).total_seconds(),0)}"
                        avg_prec = np.mean([config.get_prec(benchmark)[x] for x in datafiles])
                        items[17] = f"{avg_prec:.6f}" 
                        avg_caltime = np.mean([config.get_caltime(benchmark)[x] for x in datafiles])
                        items[18] = f"{avg_caltime}"
        
            if config.doAblation and len(items[14]) > 0 and len(items[15]) < 3:
                multiEval.flagSet.setFromZipped(items[14])
                multiEval.isRunAblative = "Y"
                ablations = multiEval.flagSet.getAblations()
                percentages = []
                defaultRuntime = float(items[11])
                bestRuntime = float(items[12])

                # recompute best, improve
                #if tuner is not None:
                #    bestRuntime = multiEval.runAll(items[14])
                #    items[12] = f"{bestRuntime:.6f}"
                #    items[13] = f"{tuner.default_perf/bestRuntime:.6f}"

                keyList = sorted(ablations.keys())
                keyList.remove(flagReader.standardOptKey)
                for key in keyList:
                    multiEval.aFlag = key
                    runtime = multiEval.runAll(ablations[key])
                    percentage = 1 - ((defaultRuntime - runtime) / (defaultRuntime - bestRuntime))
                    percentages.append(percentage)
                    outToFileAndScreen(f"Testing ablation {key} = {ablations[key]} runtime = {runtime:.6f} best = {bestRuntime:.6f} default = {defaultRuntime:.6f} percentage = {percentage:.3f}")
                
                data = np.array(percentages)
                ind = np.argsort(data)
                items[15] = f"{keyList[ind[-1]]} {percentages[ind[-1]]:.3f} {keyList[ind[-2]]} {percentages[ind[-2]]:.3f} {keyList[ind[-3]]} {percentages[ind[-3]]:.3f} {keyList[ind[-4]]} {percentages[ind[-4]]:.3f} {keyList[ind[-5]]} {percentages[ind[-5]]:.3f}"
                outToFileAndScreen(f"Top ablations = {items[15]}")
                
            lines[i] = ",".join(items)
            with open(config.queue, "w") as data:
                data.write("\n".join(lines))

if __name__ == "__main__":
    # determine hostname
    commands = "hostname"
    handle = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)
    handle.wait()
    configFileName = handle.stdout.read().decode('ascii').strip() + ".ini"
    
    config = Configuration(configFileName)
    config.read()

    newMode = config.mode
    if (len(sys.argv) > 1):
        for arg in sys.argv[1:]:
            if arg[0] == "-":
                newMode = arg[1:]
    
    if newMode != config.mode or newMode not in config.queue:
        config.mode = newMode
        config.queue = f"queue.{newMode}.{config.get_uid()}.csv"
        config.write()
    
    makeJtime()
    flagReader = FlagReader(config.gccVersion)

    # verify gcc version
    commands = f"command -v gcc-{config.gccVersion}"
    handle = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)
    handle.wait()
    results = handle.stdout.read().decode('ascii').strip()
    if len(results) == 0:
        sys.exit(f"error: gcc-{config.gccVersion} is not recognized")

    if config.mode == "default":
        if not os.path.isfile(config.queue):
            with open(config.queue, "a") as data:
                data.write("GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME\n")
                for i in range(3):
                    for gcc in config.supportedGccVersions:
                        for tuner in config.supportedTuners:
                            for benchmark in sorted(config.supportedBenchmarks.keys()):
                                datafiles = config.supportedBenchmarks[benchmark]
                                data.write(f"{gcc},{tuner},{benchmark},{json.dumps(datafiles).replace(',',':')},1000,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,False,False,,,,,,,,\n")
                                data.write(f"{gcc},{tuner},{benchmark},{json.dumps(datafiles).replace(',',':')},1000,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,True,True,,,,,,,,\n")
        processQueue(config, flagReader)
    elif config.mode == "bench":
        if not os.path.isfile(config.queue):
            with open(config.queue, "a") as data:
                data.write("GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME\n")
                for gcc in config.supportedGccVersions:
                    for p in range(1,11):
                        finfoFactor = pow(2,p)
                        for benchmark in sorted(config.supportedBenchmarks.keys()):
                            for item in config.supportedBenchmarks[benchmark]:
                                datafiles = [item]                                
                                data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},300,{finfoFactor},{config.defaultPrecTarget},6,True,True,True,,,,,,,,\n")
        processQueue(config, flagReader)
    elif config.mode == "flag":
        if not os.path.isfile(config.queue):
            with open(config.queue, "a") as data:
                data.write("GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME\n")
                for i in range(5):
                    for gcc in config.supportedGccVersions:
                        for benchmark in sorted(config.supportedBenchmarks.keys()):
                            datafiles = [1]
                            data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},100,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,False,False,,,,,,,,\n")
                            data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},100,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,False,True,,,,,,,,\n")
                            data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},100,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,True,False,,,,,,,,\n")
                            data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},100,{config.defaultPrecFinfoMult},{config.defaultPrecTarget},{config.defaultPrecMaxRuntime},True,True,True,,,,,,,,\n")
        processQueue(config, flagReader)
    elif config.mode == "finfo":
        if not os.path.isfile(config.queue):
            with open(config.queue, "a") as data:
                data.write("GCC,TUNER,BENCHMARK,DATAFILES,ITERATIONS,FINFOF,TARGETP,MAXRT,USEA,USEB,USEI,DEFAULT,BEST,IMPROVE,FLAGS,ABLATION,RUNTIME,AVG_PREC,AVG_CALTIME\n")
                for gcc in config.supportedGccVersions:
                    for benchmark in sorted(config.supportedBenchmarks.keys()):
                        for item in config.supportedBenchmarks[benchmark]:
                            for i in range(2, 12, 2):
                                datafiles = [item]
                                data.write(f"{gcc},random,{benchmark},{json.dumps(datafiles).replace(',',':')},300,{i * 1.0},{config.defaultPrecTarget},{i * config.defaultPrecMaxRuntime},True,False,False,,,,,,,,\n")
        processQueue(config, flagReader)
    elif config.mode == "recalibrate":
        OUTFILE_NAME = f"recalibrate.output.txt"
        DATAFILE_NAME = f"recalibrate.data.csv"
        for i in range(5):
            for gcc in config.supportedGccVersions:
                config.gccVersion = gcc
                config.precFinfoMult = config.defaultPrecFinfoMult
                config.precTarget = config.defaultPrecTarget
                config.precMaxRuntime = config.defaultPrecMaxRuntime
                for benchmark in sorted(config.supportedBenchmarks.keys()):
                    for item in config.supportedBenchmarks[benchmark]:
                        config.set_prec(benchmark,item,0)
                        path = f"../benchmarks/cBench/{benchmark}/src"
                        calibrateSome(path, flagReader, benchmark, [item])
    elif config.mode == "clock":
        OUTFILE_NAME = f"clock.output.txt"
        DATAFILE_NAME = f"clock.data.csv"
        for thread_iter in range(config.clockMaxThreads):
            for sample_iter in range(config.clockSamples):
                outToFileAndScreen(f"=== Clock Results:{thread_iter},{sample_iter} ===")
                for benchmark in sorted(config.supportedBenchmarks.keys()):
                    datafiles = config.supportedBenchmarks[benchmark]
                    path = f"../benchmarks/cBench/{benchmark}/src"
                    multiEval = cBenchEvaluator(path, benchmark, datafiles, flagReader, 0)
                    multiEval.num_loops = [1 for x in multiEval.parameters]
                    multiEval.num_finfos = [1 for x in multiEval.parameters]
                    multiEval.num_threads = thread_iter+1
                    outToFileAndScreen(f"{datetime.now()}: Clocking {benchmark}")
                    multiEval.calibrateParameterPre({})
                    for i in range(len(multiEval.parameters)):
                        outToFileAndScreen(f"{datetime.now()}: Threads {thread_iter+1},Sample {sample_iter+1},Dataset {i+1} / {len(multiEval.parameters)}")
                        perf = multiEval.calibrateParameter(i)
                        outToFileAndScreen(f"{datetime.now()}: Proof Loops = {multiEval.num_loops[i]}, Performance = {perf:.6f}")
                        while multiEval.num_loops[i] <= config.clockMaxLoops:
                            multiEval.num_loops[i] *= config.clockLoopFactor
                            perf = multiEval.calibrateParameter(i)
                            outToFileAndScreen(f"{datetime.now()}: Clock Loops = {multiEval.num_loops[i]}, Performance = {perf:.6f}")


