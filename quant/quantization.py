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

import numpy as np
from typing import Literal

OPTIMIZE_FAST = "O1"
OPTIMIZE_STANDARD = "O2"
OPTIMIZE_THOROUGH = "O3"

WEIGHT_INT8_VALUES = (
    ((np.arange(-128,128)/127)*np.abs(np.arange(-128,128)/127)+2*np.arange(-128,128)/127)/3
).astype(np.float32)
WEIGHT_INT6_VALUES = (
    ((np.arange(-32,32)/31)*np.abs(np.arange(-32,32)/31)+2*np.arange(-32,32)/31)/3
).astype(np.float32)
WEIGHT_INT5_VALUES = (
    ((np.arange(-16,16)/15)*np.abs(np.arange(-16,16)/15)+2*np.arange(-16,16)/15)/3
).astype(np.float32)
WEIGHT_INT4_VALUES = (
    ((np.arange(-8,8)/7)*np.abs(np.arange(-8,8)/7)+2*np.arange(-8,8)/7)/3
).astype(np.float32)
# Maps a value containing 2 weights straight into 2 decoded floats. This saves a lot of time.
WEIGHT_INT8_8_VALUES = np.array([
    WEIGHT_INT8_VALUES.view(np.uint32).astype(np.uint64)[i % 256] | \
    WEIGHT_INT8_VALUES.view(np.uint32).astype(np.uint64)[i >> 8] << 32 \
        for i in range(65536)
])
WEIGHT_INT6_8_VALUES = np.array([
    WEIGHT_INT6_VALUES.view(np.uint32).astype(np.uint64)[i % 64] | \
    WEIGHT_INT6_VALUES.view(np.uint32).astype(np.uint64)[i >> 6] << 32 \
        for i in range(4096)
])
WEIGHT_INT5_8_VALUES = np.array([
    WEIGHT_INT5_VALUES.view(np.uint32).astype(np.uint64)[i % 32] | \
    WEIGHT_INT5_VALUES.view(np.uint32).astype(np.uint64)[i >> 5] << 32 \
        for i in range(1024)
])
WEIGHT_INT4_8_VALUES = np.array([
    WEIGHT_INT4_VALUES.view(np.uint32).astype(np.uint64)[i % 16] | \
    WEIGHT_INT4_VALUES.view(np.uint32).astype(np.uint64)[i >> 4] << 32 \
        for i in range(256)
])

WEIGHT_INT4_2D_VALUES = np.empty((1024), dtype=np.uint64)
for i in range(1024):
    magnitude, angle, spin = ((i % 256 + 16) // 17, (i % 256 + 16) % 17, i >> 8)
    angle = (angle + magnitude / 2 + spin / 4) / (17 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 15).astype(np.float32).view(np.uint32).astype(np.uint64)
    b = (np.sin(angle) * magnitude / 15).astype(np.float32).view(np.uint32).astype(np.uint64)
    WEIGHT_INT4_2D_VALUES[i] = a | (b << 32)

WEIGHT_INT3_2D_VALUES = np.empty((256), dtype=np.uint64)
for i in range(256):
    magnitude, angle, spin = ((i % 64 + 8) // 9, (i % 64 + 8) % 9, i >> 6)
    angle = (angle + magnitude / 2 + spin / 4) / (9 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 7).astype(np.float32).view(np.uint32).astype(np.uint64)
    b = (np.sin(angle) * magnitude / 7).astype(np.float32).view(np.uint32).astype(np.uint64)
    WEIGHT_INT3_2D_VALUES[i] = a | (b << 32)

WEIGHT_INT2_2D_VALUES = np.empty((256), dtype=np.uint64)
for i in range(256):
    magnitude, angle, spin = ((i % 16 + 4) // 5, (i % 16 + 4) % 5, i >> 6)
    angle = (angle + magnitude / 2 + spin / 4) / (5 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 3).astype(np.float32).view(np.uint32).astype(np.uint64)
    b = (np.sin(angle) * magnitude / 3).astype(np.float32).view(np.uint32).astype(np.uint64)
    WEIGHT_INT2_2D_VALUES[i] = a | (b << 32)

# (np.arange(0,256, dtype=np.uint32)+(127-(1<<e-1)<<m)+1<<(23-m)).view(np.float32)
SCALE_UE5M3_VALUES = (np.arange(0,256, dtype=np.uint32)+889<<20).view(np.float32)

SUB_SCALES_INT4_BINS = 16
SUB_SCALES_INT2_BINS = 4
SCALE_INT4_VALUES = np.arange(1, 17, dtype=np.float32) / 16
SCALE_INT2_VALUES = np.arange(1, 5, dtype=np.float32) / 4
# Same trick as the WEIGHT_INT4_8_VALUES but for the subscales
SCALE_INT4_8_VALUES = np.array([
    SCALE_INT4_VALUES.view(np.uint32).astype(np.uint64)[i % 16] | \
    SCALE_INT4_VALUES.view(np.uint32).astype(np.uint64)[i >> 4] << 32 \
        for i in range(256)
])
SCALE_INT2_8_VALUES = np.array([
    SCALE_INT2_VALUES.view(np.uint32).astype(np.uint64)[i % 4] | \
    SCALE_INT2_VALUES.view(np.uint32).astype(np.uint64)[(i >> 2) % 4] << 32 \
        for i in range(256)
])

NUMBER_SPINS = 4

class QuantConfig:
    def __init__(
            this, 
            alg: Literal[
                "Q8L", "Q8M", "Q8S", 
                "Q6L", "Q6M", "Q6S", 
                "Q5L", "Q5M", "Q5S", 
                "Q4L", "Q4M", "Q4S", 
                "Q3L", "Q3M", "Q3S"
            ]
        ):
        this.alg = alg
        this._BITS = int(this.alg[1])
        this._ELEMENT_SIZE = this._BITS + {"L": 0.5625, "M": 0.3125, "S": 0.1875}[this.alg[2]]
        this._NONLINEAR = this._BITS >= 5
        this._SUPER_BLOCK = 128
        this._SUB_BLOCK = {"L": 8, "M": 16, "S": 32}[this.alg[2]]
        this._SUB_BLOCKS_PER_SUPER = this._SUPER_BLOCK // this._SUB_BLOCK

def dequantize_nonlinear(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    
    if type(quants) is not np.ndarray: # Convert to ndarray if needed
        quants = np.array(quants, dtype=np.uint8)
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config._ELEMENT_SIZE)

    # Unpack scales
    superBlocks = dims // config._SUPER_BLOCK
    subBlocks = dims // config._SUB_BLOCK
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subScales = quants[..., -(subBlocks // 2):]

    superScales = SCALE_UE5M3_VALUES[superScales]
    subScales = SCALE_INT4_8_VALUES[subScales].view(np.float32)
    # Use inplace reshape to merge both scale levels
    mergedScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
    mergedScales *= superScales.reshape(-1, 1)
    
    # Unpack weights
    eighth = superBlocks * config._SUPER_BLOCK // 8
    if config._BITS == 8:
        quants = WEIGHT_INT8_8_VALUES[quants[..., :eighth * 8].view(np.uint16)].view(np.float32)
    elif config._BITS == 6:
        scratch = np.empty(quants[..., eighth * 4:eighth * 6].shape, dtype=np.uint8)
        unpacked = quants[..., :eighth * 4].astype(np.uint16)
        unpacked <<= 4
        unpacked[..., :eighth * 2] |= np.bitwise_and(quants[..., eighth * 4:eighth * 6], 0xF, out=scratch)
        unpacked[..., eighth * 2:] |= np.right_shift(quants[..., eighth * 4:eighth * 6], 4, out=scratch)
        quants = WEIGHT_INT6_8_VALUES[unpacked].view(np.float32)
    elif config._BITS == 5:
        unpacked = quants[..., :eighth * 4].astype(np.uint16)
        unpacked <<= 2
        scratch = np.empty(quants[..., :eighth].shape, dtype=np.uint8)
        rest = quants[..., eighth * 4:eighth * 5]
        unpacked[..., :eighth * 1] |= np.right_shift(rest, 6, out=scratch)
        unpacked[..., eighth * 1:eighth * 2] |= \
            np.bitwise_and(np.right_shift(rest, 4, out=scratch), 3, out=scratch)
        unpacked[..., eighth * 2:eighth * 3] |= \
            np.bitwise_and(np.right_shift(rest, 2, out=scratch), 3, out=scratch)
        unpacked[..., eighth * 3:] |= np.bitwise_and(rest, 3, out=scratch)
        quants = WEIGHT_INT5_8_VALUES[unpacked].view(np.float32)
    elif config._BITS == 4:
        quants = quants[..., :eighth * 4]
        quants = WEIGHT_INT4_8_VALUES[quants].view(np.float32)

    # Use inplace reshape to multiply the elements by their respective scale
    reshaped = quants.reshape((-1, config._SUB_BLOCK))
    reshaped *= subScales.reshape((-1, 1))
    return quants

def dequantize_2d(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    
    if type(quants) is not np.ndarray: # Convert to ndarray if needed
        quants = np.array(quants, dtype=np.uint8)
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config._ELEMENT_SIZE)

    # Unpack scales
    superBlocks = dims // config._SUPER_BLOCK
    subBlocks = dims // config._SUB_BLOCK
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subInfo = quants[..., -(subBlocks // 2):]

    # Unpack spins
    spins = np.empty((*quants.shape[:-1], subBlocks), dtype=np.uint8 if config._BITS < 4 else np.uint16)
    np.left_shift(subInfo, 2, out=spins[..., :spins.shape[-1] // 2])
    spins[..., spins.shape[-1] // 2:] = subInfo
    spins &= 0xC0

    # Unpack weights
    eighth = dims // 8
    if config._BITS == 4:
        spins <<= 2
        tmp = quants[..., :eighth * 4].astype(np.uint16)
        spinMerge = tmp.reshape(-1, config._SUB_BLOCK // 2)
        spinMerge |= spins.reshape(-1, 1)
        quants = WEIGHT_INT4_2D_VALUES[tmp].view(np.float32)

    elif config._BITS == 3:
        # Unpack the 3 bit weights into a single buffer, with 6 bit integers 
        # representing 2 weights, without using temporary arrays
        unpacked = np.empty((*quants.shape[:-1], dims // 2), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 3], 0b11, out=unpacked[..., :eighth * 3])
        unpacked[..., eighth:eighth * 2] <<= 2
        unpacked[..., :eighth] |= unpacked[..., eighth:eighth * 2]
        unpacked[..., eighth * 2:eighth * 3] <<= 4
        unpacked[..., :eighth] |= unpacked[..., eighth * 2:eighth * 3]
        np.right_shift(quants[..., :eighth * 3], 2, out=unpacked[..., eighth:])
        spinMerge = unpacked.reshape(-1, config._SUB_BLOCK // 2)
        spinMerge |= spins.reshape(-1, 1)
        quants = WEIGHT_INT3_2D_VALUES[unpacked].view(np.float32)
        
    elif config._BITS == 2:
        unpacked = np.empty((*quants.shape[:-1], dims // 2), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 2], 0xF, out=unpacked[..., :unpacked.shape[-1]//2])
        np.right_shift(quants[..., :eighth * 2], 4, out=unpacked[..., unpacked.shape[-1]//2:])
        spinMerge = unpacked.reshape(-1, config._SUB_BLOCK // 2)
        spinMerge |= spins.reshape(-1, 1)
        quants = WEIGHT_INT2_2D_VALUES[unpacked].view(np.float32)

    # Unpack scales and merge them
    superScales = SCALE_UE5M3_VALUES[superScales]
    subScales = SCALE_INT2_8_VALUES[subInfo].view(np.float32)
    mergedScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
    mergedScales *= superScales.reshape(-1, 1)

    # Use inplace reshape to multiply the elements by their respective scale
    reshaped = quants.reshape(-1, config._SUB_BLOCK)
    reshaped *= mergedScales.reshape(-1, 1)
    return quants

def dequantize(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    """
    Returns a quantized embedding vector back to floats
    Params:
        quants: quantized vector or matrix
        config: configuration used for quantizing the data
    """
    if config._NONLINEAR:
        return dequantize_nonlinear(quants, config)
    return dequantize_2d(quants, config)

def quantize_nonlinear(
        data: list[float] | np.ndarray, config: QuantConfig, optimize: str = OPTIMIZE_FAST, calibration: np.ndarray | None = None
    ) -> np.ndarray:

    # Convert input data into numpy array if needed
    if type(data) is list:
        data = np.array(data, dtype=np.float32)
    # For speed; float64 obviously has half the SIMD 
    # throughput and float16 is not natively supported by most CPUs
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    inputShape = list(data.shape)
    if data.ndim != 1:
        data = data.flatten()
        if calibration is not None:
            calibration = calibration.flatten()

    # Variables initialization
    qtypeMin = -(1 << config._BITS - 1)
    qtypeMax = (1 << config._BITS - 1) - 1
    numSuperBlocks = len(data) // config._SUPER_BLOCK
    numSubBlocks = len(data) // config._SUB_BLOCK
    superBiases = np.full((numSuperBlocks), 0.0, dtype=np.float32)
    subBiases = np.full((numSubBlocks), {8: 0.45, 6: 0.35, 5: 0.2, 4: 0.0}[config._BITS], dtype=np.float32)
    superBlockErrors = [1e10] * numSuperBlocks

    def pack(data: np.ndarray, bits: int) -> np.ndarray:
        if bits == 8:
            return data
        if bits == 6:
            evenWeights = data[..., 0::2]
            oddWeights = data[..., 1::2]
            data = (oddWeights << 2) | (evenWeights >> 4)
            evenWeights &= 0xF
            rest = evenWeights[..., :evenWeights.shape[-1]//2] | (evenWeights[..., evenWeights.shape[-1]//2:] << 4)
            return np.concatenate([data, rest], axis=-1)
        if bits == 5:
            evenWeights = data[..., 0::2]
            oddWeights = data[..., 1::2]
            data = (oddWeights << 3) | (evenWeights >> 2)
            evenWeights &= 0b11
            fourth = evenWeights.shape[-1] // 4
            rest = (evenWeights[..., :fourth] << 6) | (evenWeights[..., fourth:fourth*2] << 4) | \
                (evenWeights[..., fourth*2:fourth*3] << 2) | (evenWeights[..., fourth*3:])
            return np.concatenate([data, rest], axis=-1)
        if bits == 4:
            evenWeights = data[..., 0::2]
            oddWeights = data[..., 1::2]
            oddWeights <<= 4
            evenWeights |= oddWeights
            return evenWeights
        if bits == 3:
            return pack(data[..., 0::2] | (data[..., 1::2] << 3), 6)
        if bits == 2:
            fourth = data.shape[-1] // 4
            return (data[..., :fourth] << 6) | (data[..., fourth:fourth*2] << 4) | \
                (data[..., fourth*2:fourth*3] << 2) | (data[..., fourth*3:])
        
    def combine(
            quants: np.ndarray, superScales: np.ndarray, subScales: np.ndarray
        ) -> np.ndarray:
        # Fit everything into their respective uint types
        quants = quants.astype(np.uint8)
        superScales = superScales.astype(np.uint8)
        subScales = subScales.astype(np.uint8)
        quants += 1 << config._BITS - 1
        subScales -= 1
        return np.concatenate([pack(quants, config._BITS), superScales, pack(subScales, 4)], axis=-1)
    
    def quantize_internal_step(
            data: np.ndarray, superBias: np.ndarray, subBias: np.ndarray, tiling: int = 1
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Constants
        e = 5
        m = 8 - e
        # Scales are the max absolute value of each block. If we are using tiling
        # to test multiples biases for optimization we know that the maximums will 
        # also be tiled.
        absolutes = np.abs(data)
        subMaximums = absolutes.reshape((-1, config._SUB_BLOCK))
        subMaximums = np.max(subMaximums, axis=1)
        superMaximums = subMaximums.reshape((-1, config._SUB_BLOCKS_PER_SUPER))
        superMaximums = np.max(superMaximums, axis=1)
        if tiling != 1:  # Tile the maximums and input data
            subMaximums = np.tile(subMaximums, tiling)
            superMaximums = np.tile(superMaximums, tiling)
        
        # Convert super maximums into FP8 super scales
        superScales = (superMaximums.view(np.int32) >> (23 - m)) - (111 << m)
        superScales += superBias.astype(np.int32)
        superScales = np.clip(superScales, 0, (1<<e+m)-1, out=superScales)

        # Dequantize scales and normalize maximums for the second level of scales
        unpackedScales = SCALE_UE5M3_VALUES[superScales]

        # Second level quantization
        unpackedScales /= SUB_SCALES_INT4_BINS
        normSubMaximums = subMaximums.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
        normSubMaximums /= unpackedScales.reshape(-1, 1)
        subMaximums += subBias
        np.rint(subMaximums, out=subMaximums)
        subScales = np.clip(subMaximums, 1, SUB_SCALES_INT4_BINS, out=subMaximums)

        # Mix both scales for weight normalization
        mergedScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
        mergedScales = mergedScales * unpackedScales.reshape(-1, 1)
        mergedScales = mergedScales.reshape(subScales.shape)

        # Normalize weights for later quantization
        if tiling != 1:
            quants = np.tile(data, tiling).reshape(-1, config._SUB_BLOCK)
            quants = np.divide(quants, mergedScales.reshape(-1, 1), out=quants)
            quants = np.abs(quants, out=quants)
        else:
            quants = absolutes.reshape(-1, config._SUB_BLOCK)
            quants = np.divide(quants, mergedScales.reshape(-1, 1), out=quants)
        quants = quants.reshape((data.shape[0] * tiling))
        
        # Final weights are quantized using a non linear function
        #quants = np.round(np.copysign(np.sqrt(np.abs(quants) * 3 + 1) - 1, quants) * qtypeMax)
        quants *= 3 * (qtypeMax ** 2)
        quants += qtypeMax ** 2
        np.sqrt(quants, out=quants)
        quants -= qtypeMax
        np.copysign(quants.reshape(tiling, -1), data, out=quants.reshape(tiling, -1))
        np.rint(quants, out=quants)
        return quants, superScales, subScales, mergedScales

    # Doing a round() for the super scales is not always optimal. Try
    # to find a better quantization by testing different biases for it.
    if optimize != OPTIMIZE_FAST:

        superBiasRange = {
            OPTIMIZE_STANDARD: [None, None, (  1, 3), ( 0, 2), ( 0, 1), ( 0, 1), ( 0, 1), None, ( 0, 1)],
            OPTIMIZE_THOROUGH: [None, None, (-12,12), (-2, 2), (-2, 3), (-1, 4), (-1, 4), None, ( 0, 5)],
        }[optimize][config._BITS]
        subBiasRange = {
            OPTIMIZE_STANDARD: [None, None, ( -4, 1), (-2, 1), (-3, 2), (-1, 4), (-1, 4), None, ( 0, 5)],
            OPTIMIZE_THOROUGH: [None, None, (-16,16), (-7, 6), (-3, 5), (-1, 8), (-1, 8), None, ( 0,10)],
        }[optimize][config._BITS]
        
        superBiasLength = superBiasRange[1] - superBiasRange[0]
        subBiasLength = subBiasRange[1] - subBiasRange[0]

        testSuperBias = np.repeat(np.arange(superBiasRange[0], superBiasRange[1], 1, dtype=np.float32), subBiasLength * numSuperBlocks)
        testSubBias = np.tile(np.repeat(np.arange(subBiasRange[0], subBiasRange[1], 1, dtype=np.float32), numSubBlocks), superBiasLength)

        # Common computations when testing signs
        quants, superScales, subScales, mergedScales = quantize_internal_step(
            data = data,
            superBias = testSuperBias, 
            subBias = testSubBias,
            tiling = subBiasLength * superBiasLength
        )
        
        # Dequantization computations
        quants = np.clip(quants, qtypeMin, qtypeMax, out=quants)
        quants *= 1 / ((1 << config._BITS - 1) - 1)
        # Unpack weights. x^2 + 2x keeping sign
        dequantized = np.abs(quants)
        np.multiply(dequantized, quants, out=dequantized)
        np.add(dequantized, quants, out=dequantized)
        np.add(dequantized, quants, out=dequantized)
        dequantized = dequantized
        
        # Finish by applying the scales
        mergedScales *= 0.33333
        output = dequantized.reshape(-1, config._SUB_BLOCK)
        output *= mergedScales.reshape(-1, 1)

        mse = dequantized.reshape(-1, len(data))
        np.square(np.subtract(mse, data, out=mse), out=mse)
        if calibration is not None:
            calibrated = mse.reshape(-1, len(calibration))
            calibrated *= calibration
        
        # Find optimal block roundings
        # Test all subbiases at the same to reduce computation time
        mse = mse.reshape(-1, config._SUB_BLOCK).sum(axis=1)
        mse = mse.reshape((superBiasLength, subBiasLength, numSubBlocks))
        # Find which bias had the lowest error for each subblock in each superblock
        optimalSubBiases = np.argmin(mse, axis=1)
        optimalSubBiases += subBiasRange[0]
        # With the optimal subblock rounding found, calculate the error for each superblock bias
        biasMse = np.min(mse, axis=1).reshape((superBiasLength, numSuperBlocks, -1)).sum(axis=2)

        for biasIdx in range(superBiasLength):
            # Check errors per superblock
            for block in range(numSuperBlocks):
                if biasMse[biasIdx][block] < superBlockErrors[block]:
                    # If the error with this superbias is lower overwrite
                    superBiases[block] = biasIdx + superBiasRange[0]
                    superBlockErrors[block] = biasMse[biasIdx][block]
                    # Overwrite subbiases
                    start = block * config._SUB_BLOCKS_PER_SUPER
                    end = start + config._SUB_BLOCKS_PER_SUPER
                    subBiases[start:end] = optimalSubBiases[biasIdx][start:end]

    quants, superScales, subScales, _ = quantize_internal_step(data, superBiases, subBiases)
    np.clip(quants, qtypeMin, qtypeMax, out=quants)

    # Combine all components of quantization and 
    # reshape to the expected output shape
    inputShape[-1] = -1
    return combine(
        quants.reshape(inputShape), 
        superScales.reshape(inputShape),
        subScales.reshape(inputShape)
    )

def quantize_2d(
        data: list[float] | np.ndarray, config: QuantConfig, optimize: str = OPTIMIZE_FAST, calibration: np.ndarray | None = None
    ) -> np.ndarray:

    # Convert input data into numpy array if needed
    if type(data) is list:
        data = np.array(data, dtype=np.float32)
    # For speed; float64 obviously has half the SIMD 
    # throughput and float16 is not natively supported by most CPUs
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    inputShape = list(data.shape)
    axisCalibration = None
    if data.ndim != 1:
        data = data.flatten()
        if calibration is not None:
            calibration = calibration.flatten()
            axisCalibration = (calibration[0::2], calibration[1::2])

    # Other variables initialization
    bitValues = 1 << config._BITS
    numSuperBlocks = len(data) // config._SUPER_BLOCK
    numSubBlocks = len(data) // config._SUB_BLOCK
    superBiases = np.full((numSuperBlocks), 0.5, dtype=np.float32)
    subBiases = np.full((numSubBlocks), {4: -0.1, 3: -0.25, 2: -0.4}[config._BITS], dtype=np.float32)
    bestSpins = np.empty((numSubBlocks), dtype=np.float32)
    superBlockErrors = [1e10] * numSuperBlocks
    anglesStep = (2 * np.pi) / (bitValues + 1)  # constant scalar
    magnitudesStep = 1 / (bitValues - 1)

    # Separate values into x and y coordinates
    axis0 = data[0::2]  # x
    axis1 = data[1::2]  # y
    # Convert the 2d vectors into magnitude and angle
    angles = np.arctan2(axis1, axis0)
    magnitudes = np.square(axis0) + np.square(axis1)
    magnitudes = np.sqrt(magnitudes, out=magnitudes)
    # Precompute the cosine and sine components of the angles
    axisCos = axis0 / magnitudes
    axisSin = axis1 / magnitudes

    # Scales are the max absolute value of the magnitudes of each block. 
    subMaximums = magnitudes.reshape(-1, config._SUB_BLOCK // 2)
    subMaximums = np.max(subMaximums, axis=1)
    superMaximums = subMaximums.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
    superMaximums = np.max(superMaximums, axis=1)

    # Precompute operations for the quantization of angles and magnitudes
    angles *= 1 / anglesStep
    magnitudes *= 1 / magnitudesStep

    def pack(data: np.ndarray, bits: int) -> np.ndarray:
        if bits == 8:
            return data
        if bits == 6:
            rest = data[..., :data.shape[-1]//4]
            low = np.concatenate([rest, rest >> 2, rest >> 4], axis=-1) & 0b11
            return (data[..., data.shape[-1]//4:] << 2) | low
        if bits == 4:
            return data[..., :data.shape[-1]//2] | (data[..., data.shape[-1]//2:] << 4)
        
    def combine(
            magnitudes: np.ndarray, angles: np.ndarray, superScales: np.ndarray, subScales: np.ndarray, spins: np.ndarray,
        ) -> np.ndarray:

        # Pack quantized magnitudes and angles into the final dtype
        quants = np.maximum(magnitudes * (bitValues + 1) + angles - bitValues, 0)
        quants = quants.astype(np.uint8 if config._BITS <= 4 else np.uint16)
        subScales = (subScales - 1).astype(np.uint8)

        spins = (spins * 4).astype(np.uint8)
        packedSubScales = subScales[..., 0::2] | (subScales[..., 1::2] << 2)
        packedSubScales |= spins[..., :spins.shape[-1] // 2] << 4
        packedSubScales |= spins[..., spins.shape[-1] // 2:] << 6
        return np.concatenate([pack(quants, config._BITS * 2), superScales, packedSubScales], axis=-1).astype(np.uint8)
    
    def quantize_internal(
            magnitudes: np.ndarray, 
            angles: np.ndarray, 
            superBias: np.ndarray, 
            subBias: np.ndarray, 
            spins: np.ndarray, 
            axisCalibration: tuple[np.ndarray, np.ndarray] | None,
            tiling: int = 1
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Constants
        e = 5
        m = 8 - e
        
        # Convert super maximums into FP8 super scales
        superScales = (superMaximums.view(np.int32) >> (23 - m)) - (127 - (1 << e - 1) << m)
        if tiling != 1:
            superScales = np.tile(superScales, tiling)
        superScales += superBias.astype(np.int32)
        superScales = np.clip(superScales, 0, (1<<e+m)-1, out=superScales)

        # Dequantize scales and normalize maximums for the second level of scales
        unpackedScales = SCALE_UE5M3_VALUES[superScales]

        # Second level quantization
        subScales = subMaximums
        if tiling != 1:
            subScales = np.tile(subScales, tiling)
        unpackedScales /= SUB_SCALES_INT2_BINS
        normSubMaximums = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
        normSubMaximums /= unpackedScales.reshape(-1, 1)
        biased = subScales.reshape(-1, len(subBias))
        biased += subBias
        np.ceil(subScales, out=subScales)
        subScales = np.clip(subScales, 1, SUB_SCALES_INT2_BINS, out=subScales)

        # Mix both scales for magnitude normalization
        mergedScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER)
        mergedScales = mergedScales * unpackedScales.reshape(-1, 1)
        mergedScales = mergedScales.reshape(subScales.shape)

        # Normalize magnitudes for later quantization
        if tiling != 1:
            magnitudes = np.tile(magnitudes, tiling)
        outputShape = magnitudes.shape
        magnitudes = magnitudes.reshape(-1, config._SUB_BLOCK // 2)
        magnitudes /= mergedScales.reshape(-1, 1)
        magnitudes = magnitudes.reshape(outputShape)

        # Reference axis to find the closest point in the codebook
        reference = []
        reference.append(magnitudes.reshape(tiling, -1) * axisCos.reshape(1, -1))
        reference[0] = reference[0].reshape(magnitudes.shape)
        reference.append(magnitudes.reshape(tiling, -1) * axisSin.reshape(1, -1))
        reference[1] = reference[1].reshape(magnitudes.shape)

        # Quantize the magnitudes
        np.ceil(magnitudes, out=magnitudes)
        np.clip(magnitudes, 1, bitValues - 1, out=magnitudes)
        
        # Quantize the angles
        if tiling != 1:
            angles = np.tile(angles, tiling)
        anglesShift = magnitudes.reshape(-1, len(spins), config._SUB_BLOCK // 2) * 0.5
        anglesShift += spins.reshape(-1, 1)
        anglesShift = anglesShift.reshape(magnitudes.shape)
        angles0 = np.subtract(angles, anglesShift, out=angles)
        angles1 = np.ceil(angles0)
        np.rint(angles0, out=angles0)

        # Second part: dequantize magnitudes and angles, apply calibration and find the optimal point of the codebook
        dqAngles = np.empty(angles0.shape, dtype=np.float32)
        scratch0 = np.empty(angles0.shape, dtype=np.float32)
        scratch1 = anglesShift   # We will reuse this buffer later

        # Test point 0
        dqAngles = np.add(angles0, anglesShift, out=dqAngles)
        dqAngles *= anglesStep

        cosine, sine = np.cos(dqAngles, out=scratch0), np.sin(dqAngles, out=dqAngles)
        np.square(np.subtract(np.multiply(cosine, magnitudes, out=cosine), reference[0], out=cosine), out=cosine)
        np.square(np.subtract(np.multiply(sine, magnitudes, out=sine), reference[1], out=sine), out=sine)
        if axisCalibration is not None:
            calibrated = cosine.reshape(-1, len(axisCalibration[0]))
            calibrated *= axisCalibration[0]
            calibrated = sine.reshape(-1, len(axisCalibration[0]))
            calibrated *= axisCalibration[1]
        error0 = np.add(cosine, sine, out=scratch0)

        # Test point 1
        magnitudes -= 1
        dqAngles = np.add(angles1, anglesShift, out=dqAngles)
        dqAngles -= 0.5
        dqAngles *= anglesStep

        cosine, sine = np.cos(dqAngles, out=scratch1), np.sin(dqAngles, out=dqAngles)
        np.square(np.subtract(np.multiply(cosine, magnitudes, out=cosine), reference[0], out=cosine), out=cosine)
        np.square(np.subtract(np.multiply(sine, magnitudes, out=sine), reference[1], out=sine), out=sine)
        if axisCalibration is not None:
            calibrated = cosine.reshape(-1, len(axisCalibration[0]))
            calibrated *= axisCalibration[0]
            calibrated = sine.reshape(-1, len(axisCalibration[0]))
            calibrated *= axisCalibration[1]
        error1 = np.add(cosine, sine, out=scratch1)
        
        # Select the best point from the 2 candidates
        selector = (error1 >= error0)
        outMagnitudes = np.add(magnitudes, selector, out=magnitudes)
        outAngles = np.where(selector, angles0, angles1)
        outAngles += bitValues + 1
        outAngles = np.fmod(outAngles, bitValues + 1, out=outAngles)
        return outMagnitudes, outAngles, superScales, subScales, mergedScales
    
    # Doing a round() for the scales is not always optimal. Try
    # to find a better quantization by testing different biases for it.
    if optimize != OPTIMIZE_FAST:

        superBiasRange = {
            OPTIMIZE_STANDARD: [None, None, ( 0, 1), ( 0, 1), ( 0, 1), ( 0, 1), ( 0, 1), None, ( 0, 1)],
            OPTIMIZE_THOROUGH: [None, None, (-3, 3), (-1, 4), (-1, 4), (-1, 4), (-1, 4), None, (-1, 4)],
        }[optimize][config._BITS]
        subBiasRange = {
            OPTIMIZE_STANDARD: [None, None, (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), None, (-1, 1)],
            OPTIMIZE_THOROUGH: [None, None, (-1, 2), (-1, 2), (-1, 2), (-1, 2), (-1, 2), None, (-1, 2)],
        }[optimize][config._BITS]

        superBiasLength = superBiasRange[1] - superBiasRange[0]
        subBiasLength = subBiasRange[1] - subBiasRange[0]

        tiling = subBiasLength * NUMBER_SPINS * superBiasLength
        superBias = np.repeat(np.arange(superBiasRange[0], superBiasRange[1], 1), subBiasLength * NUMBER_SPINS * numSuperBlocks)
        subBias = np.repeat(np.arange(subBiasRange[0], subBiasRange[1], 1), numSubBlocks * NUMBER_SPINS)
        spins = np.repeat(np.arange(0.0, 1.0, 0.25, dtype=np.float32), numSubBlocks)

        outMagnitudes, outAngles, superScales, subScales, mergedScales = quantize_internal(
            magnitudes, angles, superBias, subBias, spins, axisCalibration, tiling
        )

        # Merge magnitudes, angles and spins
        # For 4 bits and up we will need a wider integer for the lookup
        lookupDtype = np.uint16 if config._BITS >= 4 else np.uint8  
        values = np.multiply(outMagnitudes, bitValues + 1, out=outMagnitudes)
        values += outAngles
        values -= bitValues
        np.maximum(values, 0, out=values)
        values = values.astype(lookupDtype)

        # Convert spins to int and combine them with the values for a table lookup
        spinsBits = [None, 256, 256, 256, 1024, 4096, 16384, None, 262144][config._BITS]
        merged = values.reshape(subBiasLength * superBiasLength, NUMBER_SPINS, -1)
        merged |= np.arange(0, spinsBits, spinsBits // 4, dtype=lookupDtype).reshape(-1, 1)

        if config._BITS == 4:
            dequantized = WEIGHT_INT4_2D_VALUES[values].view(np.float32)
        elif config._BITS == 3:
            dequantized = WEIGHT_INT3_2D_VALUES[values].view(np.float32)
        elif config._BITS == 2:
            dequantized = WEIGHT_INT2_2D_VALUES[values].view(np.float32)

        # Combine with the scales
        scaled = dequantized.reshape(-1, config._SUB_BLOCK)
        scaled *= mergedScales.reshape(-1, 1)

        # Compute quantization error
        mse = dequantized.reshape(-1, len(data))
        np.square(np.subtract(mse, data, out=mse), out=mse)
        if calibration is not None:
            calibrated = mse.reshape(-1, len(calibration))
            calibrated *= calibration
        mse = mse.reshape(-1, config._SUB_BLOCK).sum(axis=1)

        # Find optimal block roundings
        # Test all subbiases at the same to reduce computation time
        mse = mse.reshape((superBiasLength, subBiasLength * NUMBER_SPINS, numSubBlocks))
        # Find which bias had the lowest error for each subblock in each superblock
        optimalSubBiases = np.argmin(mse, axis=1)
        optimalSpins = (optimalSubBiases % NUMBER_SPINS) / NUMBER_SPINS
        optimalSubBiases //= NUMBER_SPINS
        optimalSubBiases += subBiasRange[0]
        # With the optimal subblock rounding found, calculate the error for each superblock bias
        biasMse = np.min(mse, axis=1).reshape((superBiasLength, numSuperBlocks, -1)).sum(axis=2)

        for biasIdx in range(superBiasLength):
            # Check errors per superblock
            for block in range(numSuperBlocks):
                if biasMse[biasIdx][block] < superBlockErrors[block]:
                    # If the error with this superbias is lower overwrite
                    superBiases[block] = biasIdx + superBiasRange[0] 
                    superBlockErrors[block] = biasMse[biasIdx][block]
                    # Also overwrite subbiases
                    start = block * config._SUB_BLOCKS_PER_SUPER
                    end = (block + 1) * config._SUB_BLOCKS_PER_SUPER
                    subBiases[start:end] = optimalSubBiases[biasIdx][start:end]
                    bestSpins[start:end] = optimalSpins[biasIdx][start:end]

    # If we are in the fast encoder try to find decent spins with an approximation
    else:
        calibrated = magnitudes
        if calibration is not None:
            calibrated = axisCalibration[0] + axisCalibration[1]
            calibrated *= magnitudes 

        indexes = np.argmax(calibrated.reshape(-1, config._SUB_BLOCK // 2), axis=-1)
        indexes += np.arange(0, len(angles), config._SUB_BLOCK // 2, dtype=np.int64)
        bestSpins = angles[indexes]

        scratch = np.empty(bestSpins.shape, dtype=np.float32)
        bestSpins += np.pi / anglesStep  # [-pi, pi] -> [0, 2pi]
        bestSpins -= np.floor(bestSpins, out=scratch)
        bestSpins *= 4
        bestSpins = np.rint(bestSpins, out=bestSpins)
        bestSpins *= 0.25
        bestSpins -= np.floor(bestSpins, out=scratch)
    
    # Generate final quantized vector
    outMagnitudes, outAngles, superScales, subScales, _ = quantize_internal(
        magnitudes, angles, superBiases, subBiases, bestSpins, axisCalibration
    )

    # Combine all components of quantization and 
    # reshape to the expected output shape
    inputShape[-1] = -1
    return combine(
        outMagnitudes.reshape(inputShape), 
        outAngles.reshape(inputShape),
        superScales.reshape(inputShape),
        subScales.reshape(inputShape),
        bestSpins.reshape(inputShape),
    )

def quantize(
        data: list[float] | np.ndarray, config: QuantConfig, optimize: str = OPTIMIZE_FAST, calibration: np.ndarray | None = None
    ) -> np.ndarray:
    if config._NONLINEAR:
        return quantize_nonlinear(data, config, optimize, calibration)
    return quantize_2d(data, config, optimize, calibration)