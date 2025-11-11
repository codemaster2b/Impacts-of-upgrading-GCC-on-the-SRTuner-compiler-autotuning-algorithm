import os, os.path, json, random, math, base64

class FlagInfo():
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs

class FlagReader():
    def __init__(self, gccVersion):
        self.standardOptKey = "O"
        self.searchSpace = {}
        self.implies = {}
        self.benefitsFrom = {}
        self.readFlags("gcc_opts_" + str(gccVersion) + ".txt")
        if self.standardOptKey not in self.searchSpace:
            self.searchSpace[self.standardOptKey] = FlagInfo(self.standardOptKey, ["0", "1", "2", "3"])

    def readFlags(self, fileName):
        self.searchSpace.clear()
        if os.path.isfile(fileName):
            with open(fileName, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    settings = []
                    parts = [x.strip() for x in line.split(' ') if x.strip()]
                    for i, part in enumerate(parts):
                        if part.startswith("#"):
                            break

                        tokens = part.split("=")
                        name = tokens[0]
                        values = []
                        if len(tokens) > 1:
                            values.extend(tokens[1].split(","))

                        if name.startswith("O"):
                            values.append(name[1:])
                            name = self.standardOptKey
                        elif name.startswith("-fno-") and len(name) > 5:
                            values.append("false")
                            name = "-f" + name[5:]

                        setting = name
                        if values:
                            setting += "=" + values[0]
                        else:
                            setting += "=true"

                        if i == 0:
                            settings.append(setting)
                        elif i == 1:
                            if settings[-1] in self.implies:
                                self.implies[settings[-1]].append(setting)
                            else:
                                self.implies[settings[-1]] = [setting]
                            settings.append(setting)
                        else:
                            if settings[-1] in self.benefitsFrom:
                                self.benefitsFrom[settings[-1]].append(setting)
                            else:
                                self.benefitsFrom[settings[-1]] = [setting]

                        if name in self.searchSpace:
                            self.searchSpace[name].configs.extend(values)
                        else:
                            self.searchSpace[name] = FlagInfo(name, values)

            # add true and false where applicable
            for flagInfo in self.searchSpace.values():
                if len(flagInfo.configs) == 0:
                    flagInfo.configs.append("false")
                
                if "false" in flagInfo.configs and "true" not in flagInfo.configs:
                    flagInfo.configs.append("true")
                elif "true" in flagInfo.configs and "false" not in flagInfo.configs:
                    flagInfo.configs.append("false")

    def getConfigBits(self):
        bits = []
        for key in sorted(self.searchSpace.keys()):
            count = len(self.searchSpace[key].configs)
            bits.append(math.ceil(math.log2(count)) if count > 2 else 1)
        return bits
        
class FlagSet():
    def __init__(self, reader):
        self.flagReader = reader
        self.chosen = {}
        self.implies = {}
        self.error = False
        self.useImplies = False
        self.useBenefits = False
        self.searchSpaceMapping = sorted(list(reader.searchSpace.keys()))
        
        self.chosen[self.flagReader.standardOptKey] = "3"
        default = self.getDefaultEncodings()
        keys = sorted(list(self.flagReader.searchSpace.keys()))
        for i in range(len(keys)):
            if keys[i] != self.flagReader.standardOptKey:
                self.chosen[keys[i]] = self.flagReader.searchSpace[keys[i]].configs[default[i]]

    def setChosen(self, chosen):
        for key in chosen:
            if chosen[key] in self.flagReader.searchSpace[key]:
                self.chosen[key] = chosen[key]

    def setZipped(self, zipped):
        mappedEncodings = self.unzipEncodings(zipped)
        for i in range(len(mappedEncodings)):	
            key = self.searchSpaceMapping[i]
            if mappedEncodings[i] < len(self.flagReader.searchSpace[key].configs):
                self.chosen[key] = self.flagReader.searchSpace[key].configs[mappedEncodings[i]]

    def randomize(self):
        rand = random.Random()
        for key in self.searchSpaceMapping:
            count = len(self.flagReader.searchSpace[key].configs)
            self.chosen[key] = self.flagReader.searchSpace[key].configs[rand.randrange(count)] if count > 0 else None
            #print(f"{key} = {self.chosen[key]}")
        

    @staticmethod
    def fromEncodings(reader, encodings) -> 'FlagSet':
        newFlagSet = FlagSet(reader)
        keys = sorted(list(reader.searchSpace.keys()))
        if len(keys) == len(encodings):
            for i in range(len(keys)):
                key = keys[i]
                encodingIndex = encodings[i]
                if 0 <= encodingIndex < len(newFlagSet.flagReader.searchSpace[key].configs):
                    newFlagSet.chosen[key] = newFlagSet.flagReader.searchSpace[key].configs[encodingIndex]
                else:
                    newFlagSet.error = True
        else:
            newFlagSet.error = True

        return newFlagSet

    def setFromZipped(self, compressed):
        encodings = self.unzipEncodings(compressed)
        keys = sorted(list(self.flagReader.searchSpace.keys()))
        if len(keys) == len(encodings):
            for i in range(len(keys)):
                key = keys[i]
                encodingIndex = encodings[i]
                if 0 <= encodingIndex < len(self.flagReader.searchSpace[key].configs):
                    self.chosen[key] = self.flagReader.searchSpace[key].configs[encodingIndex]
                else:
                    self.error = True
        else:
            self.error = True

    def getEncodings(self):
        keys = sorted(list(self.flagReader.searchSpace.keys()))

        # start with all benefits that apply to chosen
        changedSettings = {}
        newSettings = {}
        if self.useBenefits:
            for key in keys:
                if key in self.chosen:
                    benefitKey = f"{key}={self.chosen[key]}"
                    if benefitKey in self.flagReader.benefitsFrom:
                        for setting in self.flagReader.benefitsFrom[benefitKey]:
                            pair = setting.split("=")
                            if len(pair) >= 2:
                                newSettings[pair[0]] = pair[1]
        
        # add in stages to account for different key states, such as O=1,2,3
        while len(newSettings) > 0:
            for key in newSettings:
                changedSettings[key] = newSettings[key]
            newSettings.clear()

            for key in list(changedSettings.keys()):
                benefitKey = f"{key}={changedSettings[key]}"
                if benefitKey in self.flagReader.benefitsFrom:
                    for setting in self.flagReader.benefitsFrom[benefitKey]:
                        pair = setting.split("=")
                        if len(pair) >= 2:
                            if pair[0] not in changedSettings or changedSettings[pair[0]] != pair[1]:
                                newSettings[pair[0]] = pair[1]

        # build encodings
        encodings = []
        for key in keys:
            encoding = 0
            if key in changedSettings and changedSettings[key] in self.flagReader.searchSpace[key].configs:
                encoding = self.flagReader.searchSpace[key].configs.index(changedSettings[key])
            elif self.chosen[key] in self.flagReader.searchSpace[key].configs:
                encoding = self.flagReader.searchSpace[key].configs.index(self.chosen[key])
            encodings.append(encoding)

        return encodings

    def getDefaultEncodings(self):
        keys = sorted(list(self.flagReader.searchSpace.keys()))
        
        # start with standard optimization level
        changedSettings = {}
        newSettings = {}
        newSettings[self.flagReader.standardOptKey] = self.chosen[self.flagReader.standardOptKey]

        # add in stages to account for different key states, such as O=1,2,3
        while len(newSettings) > 0:
            for key in newSettings:
                changedSettings[key] = newSettings[key]
            newSettings.clear()

            #print(f"implied changed settings {changedSettings}")
            for key in list(changedSettings.keys()):
                impliedKey = f"{key}={changedSettings[key]}"
                if impliedKey in self.flagReader.implies:
                    for setting in self.flagReader.implies[impliedKey]:
                        pair = setting.split("=")
                        if len(pair) >= 2:
                            if pair[0] not in changedSettings or changedSettings[pair[0]] != pair[1]:
                                newSettings[pair[0]] = pair[1]

        # build encodings
        encodings = []
        for key in keys:
            encoding = 0
            if key in changedSettings and changedSettings[key] in self.flagReader.searchSpace[key].configs:
                encoding = self.flagReader.searchSpace[key].configs.index(changedSettings[key])
            encodings.append(encoding)

        return encodings

    def getCompilerOptions(self):
        keys = sorted(list(self.flagReader.searchSpace.keys()))

        # Start with the standard optimization level
        compilerOptions = f" -O{self.chosen[self.flagReader.standardOptKey]}"
        defaults = self.getDefaultEncodings()
        encodings = self.getEncodings()

        # add encodings
        for i in range(min(len(keys), len(encodings))):
            key = keys[i]
            if 0 <= encodings[i] < len(self.flagReader.searchSpace[key].configs):
                value = self.flagReader.searchSpace[key].configs[encodings[i]]
                
                #print(f"encode {key}={value} useImplies={self.useImplies} default={defaults[i]} encodings={encodings[i]}")

                if key != self.flagReader.standardOptKey and (not self.useImplies or defaults[i] != encodings[i]):
                    if value == "false":
                        compilerOptions += f" -fno-{key[2:]}"
                    elif value == "true":
                        compilerOptions += f" {key}"
                    elif value is not None and value.strip():
                        compilerOptions += f" {key}={value}"

        return compilerOptions.strip() # Return trimmed string

    def getAblations(self):
        keys = sorted(list(self.flagReader.searchSpace.keys()))
        
        ablations = {}
        defaults = self.getDefaultEncodings()
        encodings = self.getEncodings()

        for i in range(min(len(keys), len(encodings))):
            if (self.useImplies and encodings[i] != defaults[i]) or (not self.useImplies and encodings[i] != 0):
                ablatedEncoding = list(encodings) # Python list copy
                ablatedEncoding[i] = defaults[i] if self.useImplies else 0
                ablations[keys[i]] = self.zipEncodings(ablatedEncoding)

        return ablations

    @staticmethod
    def encodingsToString(encodings):
        # Use map to convert all integers to strings before joining
        return ','.join(map(str, encodings))

    @staticmethod
    def encodingsFromString(encodingString):
        encodings = []
        if encodingString is None:
             return encodings
        parts = encodingString.replace(" ", "").split(',')
        for part in parts:
            if part and part.strip():
                try:
                    value = int(part)
                    encodings.append(value)
                except ValueError:
                    pass
        return encodings

    def zipMappedString(self, encodedString):
        mappedEncodings = self.encodingsFromString(encodedString)
        encodings = self.unmapEncodings(mappedEncodings)
        return self.zipEncodings(encodings)

    def unmapEncodings(self, mappedEncodings):
        keys = sorted(list(self.flagReader.searchSpace.keys()))
        encodings = []
        if len(keys) == len(mappedEncodings):
            #translate to regular ordering
            for key in keys:
                if key in self.searchSpaceMapping:
                    index = self.searchSpaceMapping.index(key)
                    encodings.append(mappedEncodings[index])        
        return encodings

    def zipEncodings(self, encodings):
        encodingBits = self.flagReader.getConfigBits()
        byteArraySize = int(math.ceil(sum(encodingBits) / 8.0))
        bytesList = bytearray(byteArraySize)
        bitIndex = 0

        for i in range(len(encodings)):
            encoding = encodings[i]
            bits = max(1,encodingBits[i])
            for j in range(bits):
                leastBit = encoding & 1
                bytePos = bitIndex // 8
                bitPos = bitIndex % 8
                if bytePos < len(bytesList):
                    bytesList[bytePos] |= (leastBit << bitPos)

                bitIndex += 1
                encoding >>= 1

        return base64.b64encode(bytes(bytesList)).decode('utf-8')

    def unzipEncodings(self, compressed):
        encodings = []
        encodingBits = self.flagReader.getConfigBits()

        try:
            for i in range(len(compressed) % 4):
                compressed += '='
            decodedBytes = base64.b64decode(compressed)

            bitIndex = 0
            # Iterate through the number of bits expected for each encoding
            for i in range(len(encodingBits)):
                bits = encodingBits[i]
                encoding = 0
                # Reconstruct the encoding value bit by bit
                for j in range(bits):
                    bytePos = bitIndex // 8
                    bitPos = bitIndex % 8
                    leastBit = (decodedBytes[bytePos] >> bitPos) & 1
                    encoding |= (leastBit << j)
                    bitIndex += 1
                encodings.append(encoding)
                
        except (base64.binascii.Error, ValueError, TypeError):
            encodings.clear()

        return encodings
