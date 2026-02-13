"""
This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU Affero General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. 
If not, see <https://www.gnu.org/licenses/>.
"""

from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
import gguf

from typing import Any
from dataclasses import dataclass
import subprocess
import sys
import os
import json
import copy

@dataclass
class Quantization:
    name: str = "None"
    bpw: float = 0
    down: str | None = None
    up: str | None = None
    
@dataclass
class Tensor:
    name: str
    size: int
    quant: Quantization
    
# Quantization methods for each tensor
QUANTIZATIONS = {
    "IQ1_S": Quantization("IQ1_S", 1.56, None, "IQ1_M"),
    "IQ1_M": Quantization("IQ1_M", 1.75, "IQ1_S", "IQ2_XXS"),
    "IQ2_XXS": Quantization("IQ2_XXS", 2.06, "IQ1_M", "IQ2_S"),
    "IQ2_S": Quantization("IQ2_S", 2.5, "IQ2_XXS", "IQ3_XXS"),
    "IQ3_XXS": Quantization("IQ3_XXS", 3.06, "IQ2_S", "IQ3_S"),
    "IQ3_S": Quantization("IQ3_S", 3.44, "IQ3_XXS", "IQ4_XS"),
    "IQ4_XS": Quantization("IQ4_XS", 4.25, "IQ3_S", "Q5_K"),
    "Q5_K": Quantization("Q5_K", 5.5, "IQ4_XS", "Q6_K"),
    "Q6_K": Quantization("Q6_K", 6.56, "Q5_K", "Q8_0"),
    "Q8_0": Quantization("Q8_0", 8.5, "Q6_K", None)
}
LOWEST_QUANT = QUANTIZATIONS["IQ1_S"]
HIGHEST_QUANT = QUANTIZATIONS["Q8_0"]

MODE_COARSE = "mode_coarse"
MODE_MEDIUM = "mode_medium"
MODE_FINE = "mode_fine"
COARSE_EPOCHS = 3
MEDIUM_EPOCHS = 2

def merge_quants(modelReader: GGUFReader, quantReaders: dict[str, GGUFReader], output: str, tensors: list[Tensor]):
    """Uses the precomputed quantized tensors to generate a new gguf with the given quantization mix"""
    writer = GGUFWriter(output, modelReader.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    for field in modelReader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue # Handled when initializing the writer
        writer.add_key_value(
            field.name, 
            field.contents(), 
            field.types[0],
            field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        )

    tensorDict = { tensors[idx].name: tensors[idx] for idx in range(len(tensors)) }

    for idx in range(len(modelReader.tensors)):
        if modelReader.tensors[idx].name not in tensorDict:
            writer.add_tensor_info(
                modelReader.tensors[idx].name, 
                modelReader.tensors[idx].data.shape, 
                modelReader.tensors[idx].data.dtype, 
                modelReader.tensors[idx].data.nbytes, 
                modelReader.tensors[idx].tensor_type
            )
            continue
        
        quantTensor = tensorDict[modelReader.tensors[idx].name].quant.name
        writer.add_tensor_info(
            quantReaders[quantTensor].tensors[idx].name, 
            quantReaders[quantTensor].tensors[idx].data.shape, 
            quantReaders[quantTensor].tensors[idx].data.dtype, 
            quantReaders[quantTensor].tensors[idx].data.nbytes, 
            quantReaders[quantTensor].tensors[idx].tensor_type
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for idx in range(len(modelReader.tensors)):
        if modelReader.tensors[idx].name not in tensorDict:
            writer.write_tensor_data(modelReader.tensors[idx].data)
            continue
        quantTensor = tensorDict[modelReader.tensors[idx].name].quant.name
        writer.write_tensor_data(quantReaders[quantTensor].tensors[idx].data)

    writer.close()
    
def calculate_kld(modelFile: str, logitsFile: str, gpuLayers: int, ctk: str, ctv: str) -> float:
    result = subprocess.run(
        args=["./llama-perplexity", "--batch_size", "4096", "-ngl", str(gpuLayers), "-ctk", ctk, "-ctv", ctv, "-fa", "1",
              "-m", modelFile, "--kl-divergence", "--kl-divergence-base", logitsFile],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    lines = result.stdout.decode().splitlines()
    for line in lines:
        if line.startswith("Mean    KLD:"):
            return float(line[12:].split("±")[0])
    raise ValueError("Could not obtain KLD value")

def calculate_score(tensors: list[Tensor], currentBPW: float, mode: str) -> tuple[float, float]:
    newBPW = sum([tensor.size * tensor.quant.bpw for tensor in tensors]) / paramCount
    merge_quants(modelReader, quantReaders, outputFile, tensors)
    kld = calculate_kld(outputFile, calibrationFile, min(maxGPULayers, numLayers), ctk, ctv)
    score = (currentKLD - kld) / abs(newBPW - currentBPW)
    print(f"New KLD: {kld} ; New BPW: {newBPW} ; Score: {score}")
    return score

def get_logits_file(tmpDirectory: str, modelFile: str, textFile: str, name: str) -> str:
    logitsPath = tmpDirectory + name
    if not os.path.exists(logitsPath):
        print(f"Calculating {name} logits...")
        subprocess.run(
            ["./llama-perplexity", "-ngl", "0", "-m", modelFile, "-f", textFile, "--save-all-logits", logitsPath],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return logitsPath
    
def get_imatrix_file(tmpDirectory: str, modelFile: str, calibrationFile: str) -> str:
    imatrixPath = tmpDirectory + "imatrix.gguf"
    if not os.path.exists(imatrixPath):
        print("Calculating imatrix...")
        subprocess.run(
            ["./llama-imatrix", "-ngl", "0", "-m", modelFile, "-f", calibrationFile, "-o", imatrixPath],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return imatrixPath
    
def get_quants(
        tmpDirectory: str, modelFile: str, imatrixFile: str, tensors: list[Tensor], minQuant: Quantization, maxQuant: Quantization
    ) -> dict[str, GGUFReader]:
    
    # Perform initial tensor quantizations
    for quant in QUANTIZATIONS.values():
        if quant.bpw < minQuant.bpw or quant.bpw > maxQuant.bpw:
            continue
        quantFile = tmpDirectory + quant.name + ".gguf"
        if os.path.exists(quantFile):
            continue
        
        print(f"Performing quant {quant.name}")            
        args = ["./llama-quantize", "--imatrix", imatrixFile]
        for tensor in tensors:
            args.append("--tensor-type")
            # llamacpp does not want to quantize these tensors to below 3bpw
            if tensor.name == "token_embd.weight" or tensor.name == "output.weight":
                uppedQuant = quant
                while uppedQuant.bpw < 3:
                    uppedQuant = QUANTIZATIONS[uppedQuant.up]
                args.append(f"{tensor.name}={uppedQuant.name}")
            else:
                args.append(f"{tensor.name}={quant.name}")
            
        args.append(modelFile)
        args.append(quantFile)
        args.append("Q4_K") # Put anything, this arg is needed

        try: os.remove(quantFile)
        except: pass
        result = subprocess.run(args=args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) 
        if result.returncode != 0 or not os.path.exists(quantFile):
            raise Exception("Quantization error")
            
    quantReaders: dict[str, GGUFReader] = {}
    for quant in QUANTIZATIONS.values():
        if quant.bpw < minQuant.bpw or quant.bpw > maxQuant.bpw:
            continue
        quantReaders[quant.name] = GGUFReader(tmpDirectory + quant.name + ".gguf")
        quantReaders[quant.name].tensors.sort(key=lambda x: x.name)
    return quantReaders


# ------------------------------- CLI PARSING -------------------------------

if len(sys.argv) < 3:
    print("Usage: \n"
          "python ./quantize-deluxe.py [OPTIONS...] [INPUT_FILE] [OUTPUT_FILE]\n"
          "NOTE: OUTPUT_FILE will be overwritten many times, burning a lot of write cycles\n"
          "-c file: calibration file to use (default this script)\n"
          "-v file: validation file to use (default this script)\n"
          "-i file: imatrix file to use (default this script)\n"
          "-b bits: target bitrate in bits per weight (default 4.0bpw)\n"
          "-e epochs: number of epochs to train for (default 6)\n"
          "-min quant: do not use quants smaller than this\n"
          "-max quant: do not use quants bigger than this\n"
          "-tensor name=quant: force the given quant for the given tensor\n"
          "-ctk quant: k cache quantization (default f16)\n"
          "-ctv quant: v cache quantization (default f16)\n"
          "-ngl layers: number of layers to offload to GPU (default 99)\n"
          "-tmp path: directory to use for temporary files")
    exit()
    
modelFile = sys.argv[-2]
outputFile = sys.argv[-1]
calibrationFile = __file__
validationFile = __file__
imatrixFile = __file__
targetBPW = 4.0
epochs = 6
maxGPULayers = 99
tmpDirectory = f"tmp_{os.path.basename(modelFile)}/"
minQuant = LOWEST_QUANT
maxQuant = HIGHEST_QUANT
ctk = "f16"
ctv = "f16"
forceQuants = {}

for argi in range(1, len(sys.argv) - 2, 2):
    if sys.argv[argi] == "-c":
        calibrationFile = sys.argv[argi + 1]
    elif sys.argv[argi] == "-v":
        validationFile = sys.argv[argi + 1]
    elif sys.argv[argi] == "-i":
        imatrixFile = sys.argv[argi + 1]
    elif sys.argv[argi] == "-b":
        targetBPW = float(sys.argv[argi + 1])
    elif sys.argv[argi] == "-e":
        epochs = int(sys.argv[argi + 1])
    elif sys.argv[argi] == "-min":
        minQuant = QUANTIZATIONS[sys.argv[argi + 1]]
    elif sys.argv[argi] == "-max":
        maxQuant = QUANTIZATIONS[sys.argv[argi + 1]]
    elif sys.argv[argi] == "-tensor":
        values = sys.argv[argi + 1].split("=")
        forceQuants[values[0]] = values[1]
    elif sys.argv[argi] == "-ctk":
        ctk = sys.argv[argi + 1]
    elif sys.argv[argi] == "-ctv":
        ctv = sys.argv[argi + 1]
    elif sys.argv[argi] == "-ngl":
        maxGPULayers = int(sys.argv[argi + 1])
    elif sys.argv[argi] == "-tmp":
        tmpDirectory = sys.argv[argi + 1]
    else:
        print(f"Unknown option {sys.argv[argi]}")
        exit()
        

# ------------------------------- PROGRAM LOGIC -------------------------------

# Create temporary directory for data
if not os.path.exists(tmpDirectory):
    os.mkdir(tmpDirectory)
    
# Get the logits of the original model to compare and the imatrix
calibrationFile = get_logits_file(tmpDirectory, modelFile, calibrationFile, "calibration")
validationFile = get_logits_file(tmpDirectory, modelFile, validationFile, "validation")
imatrixFile = get_imatrix_file(tmpDirectory, modelFile, imatrixFile)
    
# Read tensors present in the file
tensors: list[Tensor] = []
modelReader = GGUFReader(modelFile)
modelReader.tensors.sort(key=lambda x: x.name) # Order of read tensors might change...
numLayers = -1  # Replace with num layers read from gguf
paramCount = 0

for tensor in modelReader.tensors:
    if tensor.tensor_type == gguf.constants.GGMLQuantizationType.F32:
        continue
    tensors.append(Tensor(
        name=tensor.name,
        size=tensor.n_elements,
        quant=LOWEST_QUANT, # Start at the lowest quant
    ))
    paramCount += tensor.n_elements
    if tensor.name.startswith("blk"):
        numLayers = max(numLayers, int(tensor.name.split(".")[1]) + 1)
    tensorType = ".".join(tensor.name.split(".")[-2:])
    
# Perform initial tensor quantizations
quantReaders = get_quants(tmpDirectory, modelFile, imatrixFile, tensors, minQuant, maxQuant)

# Generate initial quantization mix based on the target bpw. Start at the lowest quantization
for tensor in tensors:
    tensor.quant = minQuant
    if tensor.name == "token_embd.weight" or tensor.name == "output.weight":
        while tensor.quant.bpw < 3:
            tensor.quant = QUANTIZATIONS[tensor.quant.up]
    # Apply forced quant types
    if tensor.name in forceQuants:
        tensor.quant = QUANTIZATIONS[forceQuants[tensor.name]]
        
# Increase quantization for all tensors until we reach the target bpw
quant = minQuant
while True:
    if quant.name == maxQuant.name:
        break
    quant = QUANTIZATIONS[quant.up]
    
    newTensors = copy.deepcopy(tensors)
    for tensor in newTensors:
        if tensor.quant.bpw >= quant.bpw or tensor.name in forceQuants:
            continue
        tensor.quant = quant
        
    currentBPW = sum([tensor.size * tensor.quant.bpw for tensor in newTensors]) / paramCount
    if currentBPW > targetBPW:
        break
    tensors = newTensors
   
epochInfo: list[dict[str, Any]] = []
if os.path.exists(tmpDirectory + f"epochs_{targetBPW}.json"):
    # If there exists quant information from a previous run start from that
    epochInfo = json.load(open(tmpDirectory + f"epochs_{targetBPW}.json"))
    for tensor in tensors:
        tensor.quant = QUANTIZATIONS[epochInfo[-1]["tensors"][tensor.name]]
    del epochInfo[-1] # Will be added back at the start of the epoch

# Information of gradients for each epoch
gradInfo: list[dict[str, Any]] = []
if os.path.exists(tmpDirectory + f"gradients_{targetBPW}.json"):
    gradInfo = json.load(open(tmpDirectory + f"gradients_{targetBPW}.json", "r"))

while True:
    currentBPW = sum([tensor.size * tensor.quant.bpw for tensor in tensors]) / paramCount
    # Create the output model from the last epoch
    print(f"\nOutputting quantized model from epoch {len(epochInfo) + 1}")
    merge_quants(modelReader, quantReaders, outputFile, tensors)
    print(f"Output done")
    print(f"Testing KLD for epoch {len(epochInfo) + 1}")
    currentKLD = calculate_kld(outputFile, calibrationFile, min(maxGPULayers, numLayers), ctk, ctv)
    validationKLD = calculate_kld(outputFile, validationFile, min(maxGPULayers, numLayers), ctk, ctv)
    print(f"Current BPW: {currentBPW}\nCurrent KLD: {currentKLD}\nValidation KLD: {validationKLD}\n")

    currentEpoch = len(epochInfo)
    gradMode = MODE_COARSE if currentEpoch < COARSE_EPOCHS \
          else MODE_MEDIUM if currentEpoch < COARSE_EPOCHS + MEDIUM_EPOCHS \
          else MODE_FINE
    epochInfo.append({
        "grad": gradMode,
        "epoch": currentEpoch,
        "kld": currentKLD,
        "validation": validationKLD,
        "tensors": { tensor.name: tensor.quant.name for tensor in tensors },
    })
    json.dump(epochInfo, open(tmpDirectory + f"epochs_{targetBPW}.json", "w", encoding="utf-8"), indent=4)
    
    if len(epochInfo) > epochs:
        break

    # If for whatever reason the grad mode is changed delete the grad info for the last epoch
    if len(gradInfo) > currentEpoch and gradInfo[currentEpoch]["grad"] != gradMode:
        del gradInfo[-1]
    if len(gradInfo) <= currentEpoch:  # Add a new empty epoch if not present
        gradInfo.append({ "grad": gradMode, "up": [], "down": [] })

    # Group name as key and list of tensor names inside each group
    tensorGroups: dict[str, set[str]] = {}  
    if gradMode == MODE_COARSE:
        # ######## COARSE GRADIENTS MODE ######## #
        # Modify tensors by types, getting a first approximation
        # of what quantization each tensor should have
        for tensor in tensors:
            tensorType = ".".join(tensor.name.split(".")[-2:])
            if tensorType not in tensorGroups:
                tensorGroups[tensorType] = set()
            tensorGroups[tensorType].add(tensor.name)

    elif gradMode == MODE_MEDIUM:
        GRANULARITY = 4
        for tensor in tensors:
            tensorType = ".".join(tensor.name.split(".")[-2:])
            blockNumber = int(tensor.name.split(".")[1]) if tensor.name.startswith("blk") else 0
            groupID = f"{blockNumber // GRANULARITY}.{tensorType}"
            if groupID not in tensorGroups:
                tensorGroups[groupID] = set()
            tensorGroups[groupID].add(tensor.name)

    else:
        # ######## FINE GRADIENTS MODE ######## #
        # Modify each tensor individually, for the necessary 
        # granularity to achieve optimal quantization
        for tensor in tensors:
            tensorGroups[tensor.name] = set([tensor.name])
    
    # First find the effect of upgrading and downgrading each tensor group
    for mode in ["up", "down"]:
            
        for group in tensorGroups.keys():
            
            # Check if we already have the gradient for this tensor group
            if any([group == grad["group"] for grad in gradInfo[currentEpoch][mode]]):
                continue
            modifiedTensorNames = []
            newTensors = copy.deepcopy(tensors)
            
            for tensor in newTensors:
                if tensor.name in forceQuants or \
                    tensor.name not in tensorGroups[group] or \
                    tensor.quant.name == (maxQuant.name if mode == "up" else minQuant.name):
                    continue
                    
                newQuant = QUANTIZATIONS[getattr(tensor.quant, mode)]
                if tensor.name in ["token_embd.weight", "output.weight"] and newQuant.bpw < 3:
                    continue
                oldQuant = tensor.quant
                tensor.quant = newQuant
                modifiedTensorNames.append(tensor.name)
                
            if newTensors == tensors:  # No modified tensors, skip
                continue
            print(f"Tensor {group}, {oldQuant.name} -> {newQuant.name}")
            score = calculate_score(newTensors, currentBPW, mode)
            gradInfo[currentEpoch][mode].append({
                "tensors": modifiedTensorNames, "group": group, "score": score,
            })
            json.dump(gradInfo, open(tmpDirectory + f"gradients_{targetBPW}.json", "w", encoding="utf-8"), indent=4)
            
    # Now that we have the "steepness" for modifying each quant we have to choose 
    # which changes are the most beneficial while staying below the target bitrate.
    gradInfo[currentEpoch]["up"].sort(key=lambda x: x["score"])
    gradInfo[currentEpoch]["down"].sort(key=lambda x: x["score"])

    def mod_tensors(scoreThreshold: float = 0) -> list[Tensor]:
        """Upgrade/downgrade quality of all tensors based on the obtained scores,
        while staying below the target bitrate. Optionally stop when the upgrade
        scores drop below the threshold, which can sometimes help.
        Returns the updated tensors."""
        grads = copy.deepcopy(gradInfo[currentEpoch])
        testTensors = copy.deepcopy(tensors)
        modifiedGroups = set()
        bestTensors = copy.deepcopy(tensors)
        bestScore = 0
        score = 0

        while True:
            newBPW = sum([tensor.size * tensor.quant.bpw for tensor in testTensors]) / paramCount
            if newBPW <= targetBPW and score > bestScore:  # Record best upgrade/downgrade combination
                bestTensors = copy.deepcopy(testTensors)
                bestScore = score

            if newBPW > targetBPW and len(grads["up"]) > 0 and \
                grads["up"][-1]["score"] < scoreThreshold:
                break

            mode = "down" if newBPW >= targetBPW else "up"
            if len(grads[mode]) == 0:  # No tensors remain
                break

            grad = grads[mode].pop(-1)
            # It is possible for the same tensor to improve quality by upgrading
            # AND downgrading their quant. But we should only do one of them.
            if grad["group"] not in modifiedGroups: 
                for idx in range(len(testTensors)):
                    if testTensors[idx].name in grad["tensors"]:
                        testTensors[idx].quant = QUANTIZATIONS[getattr(testTensors[idx].quant, mode)]
                score += grad["score"]
                modifiedGroups.add(grad["group"])

        return bestTensors

    if gradMode == MODE_COARSE:
        modifiedTensors = mod_tensors(0)
        # Make sure this actually improves KLD
        merge_quants(modelReader, quantReaders, outputFile, modifiedTensors)
        newKLD = calculate_kld(outputFile, calibrationFile, min(maxGPULayers, numLayers), ctk, ctv)
        if newKLD < currentKLD:
            tensors = modifiedTensors

    else:
        bestKLD = currentKLD
        bestTensors = tensors
        # Test changing quantizations with a few different score thresholds, and keep the best one
        for threshold in [x / 8 for x in range(8)]:
            modifiedTensors = mod_tensors(gradInfo[currentEpoch]["up"][-1]["score"] * threshold)
            merge_quants(modelReader, quantReaders, outputFile, modifiedTensors)
            newKLD = calculate_kld(outputFile, calibrationFile, min(maxGPULayers, numLayers), ctk, ctv)
            if newKLD < bestKLD:
                bestKLD = newKLD
                bestTensors = modifiedTensors
        tensors = bestTensors

