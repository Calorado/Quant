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

from quantization import quantize, dequantize, QuantConfig, OPTIMIZE_FAST, OPTIMIZE_STANDARD, OPTIMIZE_THOROUGH
from quantization_cupy import cu_dequantize

import torch
import numpy as np
import cupy as cu
import psutil

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

PHYSICAL_CORES = psutil.cpu_count(logical=False)
THREAD_POOL = ThreadPoolExecutor(PHYSICAL_CORES)
# Synchronize with the NULL stream, as that is where we will create the output buffers
STREAM = cu.cuda.Stream(non_blocking=False)

numFP16Weights = 0

def quantize_matrix(input: np.ndarray, config: QuantConfig, optimize: int, calibration: np.ndarray | None = None) -> np.ndarray:
    # Quantize a 2D matrix using multithreading
    # To keep peak memory usage under control do not 
    # process more than this amount of weights at a time
    WEIGHT_BATCH_SIZE = {
        OPTIMIZE_FAST: 262144, OPTIMIZE_STANDARD: 65536, OPTIMIZE_THOROUGH: 16384
    }[optimize]
    rowsPerBatch = min(
        WEIGHT_BATCH_SIZE // input.shape[1], 
        input.shape[0] // PHYSICAL_CORES
    )
    if rowsPerBatch < 1:
        rowsPerBatch = 1
    output = np.empty(
        (
            input.shape[0], 
            int(input.shape[1] * (config.element_size()) / 8)
        ), 
        dtype=np.uint8
    )
    # Progress
    lock = Lock()
    row = [0]
    
    def quantize_thread() -> np.ndarray:
        while True:
            with lock:
                curRow = row[0]
                row[0] += rowsPerBatch
            if curRow >= output.shape[0]:
                break
            
            calibrationRows = None
            if calibration is not None:
                calibrationRows = calibration[curRow:curRow+rowsPerBatch]
            output[curRow:curRow+rowsPerBatch] = quantize(
                input[curRow:curRow+rowsPerBatch], config, optimize, calibrationRows
            )
    threads = [THREAD_POOL.submit(quantize_thread) for _ in range(PHYSICAL_CORES)]
    for thread in threads:
        thread.result()
    return output

class QuantizedEmbedding(torch.nn.Module):

    def __init__(this, embeddings: torch.nn.Embedding, quant: str = "Q4L", optimize: int = OPTIMIZE_FAST):
        """Initializes a embeddings like module with quantized weights.
        Args:
            embeddings: an existing torch.nn.Embedding module which will be quantized
            quant: method to use for quantization
            optimize: how much time to spend trying to minimize the quantization error
        """
        super().__init__()

        this.shape = embeddings.weight.shape
        this.device = embeddings.weight.device
        this.dtype = embeddings.weight.dtype
        this.quantConfig = QuantConfig(quant)
        this.quantizedWeights = quantize_matrix(embeddings.weight.cpu().detach().numpy(), this.quantConfig, optimize)
        if this.device.type == "cuda":
            this.quantizedWeights = torch.tensor(this.quantizedWeights, dtype=torch.uint8, device="cuda")

    def forward(this, indices: torch.Tensor):
        if this.device.type == "cpu":
            weights = dequantize(this.quantizedWeights[indices.detach().numpy()], this.quantConfig)
            return torch.from_numpy(weights)
        if this.device.type == "cuda":
            weights = cu_dequantize(cu.asarray(this.quantizedWeights[indices]), this.quantConfig)
            return torch.as_tensor(weights, device="cuda")
    
    def nelement(this) -> int:
        return this.shape[0] * this.shape[1]

    def element_size(this) -> float:
        return this.quantConfig.element_size() / 8
    
class CalibrationLinear(torch.nn.Module):
    """Class to record activation statistics from a torch.nn.Linear module, and
    then use those statistics as calibration data for quantization."""

    def __init__(this, module: torch.nn.Linear):
        super().__init__()
        this.module = module
        this.weights = torch.zeros(module.weight.shape[-1], dtype=torch.float32, device="cpu")
        this.numPasses = 0

    def forward(this, input: torch.Tensor):
        reducedSum = torch.square(input.type(torch.float32))
        while reducedSum.dim() != 1:
            reducedSum = torch.sum(reducedSum, axis=0)
        reducedSum = reducedSum.to("cpu")

        this.weights += torch.abs(reducedSum)
        this.numPasses += 1
        return this.module(input)
    
    def get_calibration_matrix(this) -> np.ndarray:
        if this.numPasses == 0:
            return np.tile(np.ones((1, this.module.weight.shape[-1]), dtype=np.float32), (this.module.weight.shape[0], 1))
        return (this.weights / this.numPasses).unsqueeze(0).tile((this.module.weight.shape[0], 1)).numpy()
    
    def nelement(this) -> int:
        return this.weights.shape[0] * this.weights.shape[1]
    
    def element_size(this) -> int:
        return 4
    
class QuantizedLinear(torch.nn.Module):

    def __init__(
            this, 
            module: torch.nn.Linear | QuantizedEmbedding | CalibrationLinear, 
            quant: str = "Q4L",
            optimize: int = OPTIMIZE_FAST,
            batches: int = 1,
            buffer: torch.Tensor | None = None,
            xyz = False
        ):
        """Initializes a linear like module with quantized weights.
        Args:
            module: an existing torch.nn.Linear module which will be quantized or an existing 
                QuantizedEmbedding module for models with tied embeddings
            quant: method to use for quantization. Ignored if a QuantizedEmbedding
                object is passed as module
            optimize: how much time to spend trying to minimize the quantization error. 
                Ignored if a QuantizedEmbedding object is passed as module
            batches: computing this module requires dequantizing it into a scratch buffer,
                which might take a lot of memory, negating the benefits of quantization. 
                An alternative is to divide the computation into this many equal sized steps, 
                which divides the scratch memory needed by the number of steps
            buffer: use this preallocated torch.Tensor as scratch memory for dequantization,
                which allows employing CUDA graphs for improved performance
        """
        global numFP16Weights
        super().__init__()

        this.mx = None
        this.ind = None

        if type(module) is torch.nn.Linear or type(module) is CalibrationLinear:
            calibration = None
            if type(module) is CalibrationLinear:
                calibration = module.get_calibration_matrix()
                module = module.module # Get the actual torch.nn.Linear module

            this.bias = module.bias # Usually small, not worth quantizing
            this.shape = module.weight.shape
            this.device = module.weight.device
            this.dtype = module.weight.dtype
            this.quantConfig = QuantConfig(quant)

            if False and calibration is not None:
                sortedActivations = np.flip(np.sort(calibration[0]))
                median = sortedActivations[calibration.shape[1] // 2]
                print(sortedActivations[:20])
                print(len(calibration[0]))

                this.ind = []
                this.mx = []
                while sortedActivations[len(this.ind)] > median * 60:
                #for activation in range(0):
                    this.ind.append(np.argmax(calibration[0]))
                    this.mx.append(torch.clone(module.weight[:, this.ind[-1]]))
                    calibration[:, this.ind[-1]] = 0

                print(this.ind)
                numFP16Weights += len(calibration[0]) * len(this.mx)
                print(f"FP16 {numFP16Weights}")



            this.quantizedWeights = quantize_matrix(module.weight.numpy(force=True), this.quantConfig, optimize, calibration)
            if this.device.type == "cuda":
                this.quantizedWeights = torch.tensor(this.quantizedWeights, dtype=torch.uint8, device="cuda")
        # Initialize linear layer with the weights from a QuantizedEmbeddings object. Good for models that use tied embeddings.
        else:
            this.bias = None
            this.shape = module.shape
            this.device = module.device
            this.dtype = module.dtype
            this.quantConfig = module.quantConfig
            this.quantizedWeights = module.quantizedWeights

        this.batches = 1#batches
        this.buffer = None#buffer
        """if this.buffer is not None:
            this.buffer = buffer[:this.shape[0] * this.shape[1] // this.batches].reshape((-1, this.shape[1]))
            this.buffer = cu.asarray(this.buffer) # These will share the underlying memory"""
        this.graphs = [None] * this.batches # CUDA graphs

    def forward(this, input: torch.Tensor):

        def dequantize_cpu(output: np.ndarray, quants: np.ndarray):
            rowsPerThread = output.shape[0] / PHYSICAL_CORES
            def dequantize_row(idx: int):
                srow = round(rowsPerThread * idx)
                erow = round(rowsPerThread * (idx + 1))
                if erow == srow: 
                    return
                output[srow:erow] = dequantize(quants[srow:erow], this.quantConfig)
            threads = [THREAD_POOL.submit(dequantize_row, idx) for idx in range(PHYSICAL_CORES)]
            for thread in threads: 
                thread.result()

        def dequantize_cuda(output: cu.ndarray, quants: cu.ndarray, idx: int) -> cu.ndarray:
            # Apply graphs for a massive reduction in CPU cost
            if output is not None:
                if this.graphs[idx] is None:
                    with STREAM:
                        STREAM.begin_capture()
                        cu_dequantize(cu.asarray(quants), this.quantConfig, out=output)
                        this.graphs[idx] = STREAM.end_capture()
                this.graphs[idx].launch(stream=STREAM)
                STREAM.synchronize()
                return output
            # Otherwise dequantize as usual
            return cu_dequantize(cu.asarray(quants), this.quantConfig)

        # Compute this module in a single step, as this is faster
        if this.batches == 1:
            if this.device.type == "cpu":
                weights = np.empty(this.shape, dtype=np.float32)
                dequantize_cpu(weights, this.quantizedWeights)
                return torch.nn.functional.linear(input, torch.from_numpy(weights), this.bias)
            
            if this.device.type == "cuda":
                weights = dequantize_cuda(this.buffer, this.quantizedWeights, 0)
                x = torch.as_tensor(weights, device="cuda")
                if this.mx is not None:
                    for idx in range(len(this.mx)):
                        x[:, this.ind[idx]] = this.mx[idx]
                return torch.nn.functional.linear(input, torch.as_tensor(weights, device="cuda"), this.bias)
        
        # Compute this module in multiple steps, dequantizing only a subset of the weights per step to limit peak memory
        if this.device.type == "cpu":
            rowsPerBatch = this.shape[0] // this.batches
            output = torch.empty((*input.shape[:-1], this.shape[0]), dtype=torch.float32, device="cpu")
            weights = np.empty((rowsPerBatch, this.shape[1]), dtype=np.float32)
            for batch in range(this.batches):
                srow = batch * rowsPerBatch
                erow = (batch + 1) * rowsPerBatch
                dequantize_cpu(weights, this.quantizedWeights[srow:erow])
                torch.matmul(input, torch.from_numpy(weights).T, out=output[..., srow:erow])
        
        if this.device.type == "cuda":
            rowsPerBatch = this.shape[0] // this.batches
            output = torch.empty((*input.shape[:-1], this.shape[0]), dtype=torch.float16, device="cuda")
            for batch in range(this.batches):
                srow = batch * rowsPerBatch
                erow = (batch + 1) * rowsPerBatch
                weights = dequantize_cuda(this.buffer, this.quantizedWeights[srow:erow], batch)
                torch.matmul(input, torch.as_tensor(weights, device="cuda").T, out=output[..., srow:erow])
                
        if this.bias is not None:
            output += this.bias
        return output
    
    def nelement(this) -> int:
        return this.shape[0] * this.shape[1]
    
    def element_size(this) -> float:
        return this.quantConfig.element_size() / 8