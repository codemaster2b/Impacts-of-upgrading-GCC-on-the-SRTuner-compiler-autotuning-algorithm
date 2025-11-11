import os, os.path, json, time
from datetime import datetime

class Configuration():
    def __init__(self, fileName):
        self.fileName = fileName
        self.mode = "default"
        self.gccVersion = "11"
        self.precTarget = 1.0
        self.defaultPrecTarget = 1.0
        self.precFinfoMult = 2.0
        self.defaultPrecFinfoMult = 2.0
        self.precMaxRuntime = 3.0
        self.defaultPrecMaxRuntime = 3.0
        self.clockMaxThreads = 12
        self.clockSamples = 100
        self.clockMaxLoops = 256
        self.clockLoopFactor = 2.0
        self.queue = f"queue.default.{self.get_uid()}.csv"
        self.doAblation = False
        self.doBenefits = False
        self.doCombineFairly = False
        self.doImplies = False
        self.supportedGccVersions = ["7", "8", "9", "10", "11", "12", "13", "14"]
        self.supportedTuners = ["random", "srtuner"]
        supportedBenchmarks={"automotive_bitcount": [12], "automotive_susan_c": [15], "automotive_susan_e": [3], "automotive_susan_s": [12], "bzip2e": [20], "consumer_jpeg_c": [0], "consumer_tiff2rgba": [0], "network_dijkstra": [2], "office_rsynth": [17], "security_sha": [15], "telecom_adpcm_c": [4], "telecom_adpcm_d": [16]}
        self.calibratedThreads = {}
        self.calibratedFinfos = {}
        self.calibratedLoops = {}
        self.calibratedRuntime = {}
        self.calibratedPrec = {}
        self.calibratedCaltime = {}
        self.tuner = "notuner"
        
    def get_uid(self):
        return int(datetime.today().timestamp())

    def get_threads(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key in self.calibratedThreads:
    	    return self.calibratedThreads[key]
        else:
    	    return []

    def set_threads(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key not in self.calibratedThreads:
            self.calibratedThreads[key] = []
        while len(self.calibratedThreads[key]) <= i:
            self.calibratedThreads[key].append(0)
        self.calibratedThreads[key][i] = value

    def get_finfos(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key in self.calibratedFinfos:
    	    return self.calibratedFinfos[key]
        else:
    	    return []

    def set_finfos(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key not in self.calibratedFinfos:
            self.calibratedFinfos[key] = []
        while len(self.calibratedFinfos[key]) <= i:
            self.calibratedFinfos[key].append(0)
        self.calibratedFinfos[key][i] = value

    def get_loops(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key in self.calibratedLoops:
    	    return self.calibratedLoops[key]
        else:
    	    return []

    def set_loops(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}"
        if key not in self.calibratedLoops:
            self.calibratedLoops[key] = []
        while len(self.calibratedLoops[key]) <= i:
            self.calibratedLoops[key].append(0)
        self.calibratedLoops[key][i] = value

    def get_runtime(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key in self.calibratedRuntime:
    	    return self.calibratedRuntime[key]
        else:
    	    return []

    def set_runtime(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key not in self.calibratedRuntime:
            self.calibratedRuntime[key] = []
        while len(self.calibratedRuntime[key]) <= i:
            self.calibratedRuntime[key].append(0)
        self.calibratedRuntime[key][i] = value

    def get_prec(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key in self.calibratedPrec:
    	    return self.calibratedPrec[key]
        else:
    	    return []

    def set_prec(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key not in self.calibratedPrec:
            self.calibratedPrec[key] = []
        while len(self.calibratedPrec[key]) <= i:
            self.calibratedPrec[key].append(0)
        self.calibratedPrec[key][i] = value

    def get_caltime(self, benchmark):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key in self.calibratedCaltime:
    	    return self.calibratedCaltime[key]
        else:
    	    return []

    def set_caltime(self, benchmark, i, value):
        key = f"{benchmark}:{self.precTarget:.2f}:{self.precFinfoMult:.2f}:{self.precMaxRuntime:.2f}:{self.gccVersion}"
        if key not in self.calibratedCaltime:
            self.calibratedCaltime[key] = []
        while len(self.calibratedCaltime[key]) <= i:
            self.calibratedCaltime[key].append(0)
        self.calibratedCaltime[key][i] = value

    def read(self):
        if os.path.isfile(self.fileName):
            file=open(self.fileName)
            lines = file.read().split("\n")
            for line in lines:
                segments = line.split("=")
                if segments[0] == "mode":
                    self.mode = segments[1]
                elif segments[0] == "gccVersion":
                    self.gccVersion = segments[1]
                elif segments[0] == "precTarget":
                    self.precTarget = float(segments[1])
                elif segments[0] == "precFinfoMult":
                    self.precFinfoMult = float(segments[1])
                elif segments[0] == "precMaxRuntime":
                    self.precMaxRuntime = float(segments[1])
                elif segments[0] == "clockMaxThreads":
                    self.clockMaxThreads = int(segments[1])
                elif segments[0] == "clockSamples":
                    self.clockSamples = int(segments[1])
                elif segments[0] == "clockMaxLoops":
                    self.clockMaxLoops = int(segments[1])
                elif segments[0] == "clockLoopFactor":
                    self.clockLoopFactor = float(segments[1])
                elif segments[0] == "queue":
                    self.queue = segments[1]
                elif segments[0] == "doAblation":
                    self.doAblation = bool(segments[1])
                elif segments[0] == "doBenefits":
                    self.doBenefits = bool(segments[1])
                elif segments[0] == "doCombineFairly":
                    self.doCombineFairly = bool(segments[1])
                elif segments[0] == "doImplies":
                    self.doImplies = bool(segments[1])
                elif segments[0] == "supportedGccVersions":
                    self.supportedGccVersions = json.loads(segments[1])
                elif segments[0] == "supportedTuners":
                    self.supportedTuners = json.loads(segments[1])
                elif segments[0] == "supportedBenchmarks":
                    self.supportedBenchmarks = json.loads(segments[1])
                elif segments[0] == "defaultPrecTarget":
                    self.defaultPrecTarget = float(segments[1])
                elif segments[0] == "defaultPrecFinfoMult":
                    self.defaultPrecFinfoMult = float(segments[1])
                elif segments[0] == "defaultPrecMaxRuntime":
                    self.defaultPrecMaxRuntime = float(segments[1])
                elif segments[0] == "calibratedThreads":
                    self.calibratedThreads = json.loads(segments[1])
                elif segments[0] == "calibratedFinfos":
                    self.calibratedFinfos = json.loads(segments[1])
                elif segments[0] == "calibratedLoops":
                    self.calibratedLoops = json.loads(segments[1])
                elif segments[0] == "calibratedRuntime":
                    self.calibratedRuntime = json.loads(segments[1])
                elif segments[0] == "calibratedPrec":
                    self.calibratedPrec = json.loads(segments[1])
                elif segments[0] == "calibratedCaltime":
                    self.calibratedCaltime = json.loads(segments[1])
            file.close()
    
    def write(self):
        with open(self.fileName, "w") as out:
            out.write(f"mode={self.mode}\n")
            out.write(f"gccVersion={self.gccVersion}\n")
            out.write(f"precTarget={self.precTarget}\n")
            out.write(f"precFinfoMult={self.precFinfoMult}\n")
            out.write(f"precMaxRuntime={self.precMaxRuntime}\n")
            out.write(f"clockMaxThreads={self.clockMaxThreads}\n")
            out.write(f"clockSamples={self.clockSamples}\n")
            out.write(f"clockMaxLoops={self.clockMaxLoops}\n")
            out.write(f"clockLoopFactor={self.clockLoopFactor}\n")
            out.write(f"queue={self.queue}\n")
            out.write(f"doAblation={self.doAblation}\n")
            out.write(f"doBenefits={self.doBenefits}\n")
            out.write(f"doCombineFairly={self.doCombineFairly}\n")
            out.write(f"doImplies={self.doImplies}\n")
            out.write(f"supportedGccVersions={json.dumps(self.supportedGccVersions)}\n")
            out.write(f"supportedTuners={json.dumps(self.supportedTuners)}\n")
            out.write(f"supportedBenchmarks={json.dumps(self.supportedBenchmarks)}\n")
            out.write(f"defaultPrecTarget={self.defaultPrecTarget}\n")
            out.write(f"defaultPrecFinfoMult={self.defaultPrecFinfoMult}\n")
            out.write(f"defaultPrecMaxRuntime={self.defaultPrecMaxRuntime}\n")
            out.write(f"calibratedThreads={json.dumps(self.calibratedThreads)}\n")
            out.write(f"calibratedFinfos={json.dumps(self.calibratedFinfos)}\n")
            out.write(f"calibratedLoops={json.dumps(self.calibratedLoops)}\n")
            out.write(f"calibratedRuntime={json.dumps(self.calibratedRuntime)}\n")
            out.write(f"calibratedPrec={json.dumps(self.calibratedPrec)}\n")
            out.write(f"calibratedCaltime={json.dumps(self.calibratedCaltime)}\n")
            
