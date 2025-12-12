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
from dataclasses import dataclass
from typing import Literal

OPTIMIZE_FAST = -1
OPTIMIZE_STANDARD = 0
OPTIMIZE_THOROUGH = 1

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
WEIGHT_INT3_VALUES = (
    ((np.arange(-4,4)/3)*np.abs(np.arange(-4,4)/3)+2*np.arange(-4,4)/3)/3
).astype(np.float32)
WEIGHT_INT2_VALUES = (
    ((np.arange(-2,2)/1)*np.abs(np.arange(-2,2)/1)+2*np.arange(-2,2)/1)/3
).astype(np.float32)
# Maps a value containing 2 weights straight into 2 decoded floats. This saves a lot of time.
WEIGHT_INT4_8_VALUES = np.array([
    WEIGHT_INT4_VALUES.view(np.uint32).astype(np.uint64)[i % 16] | \
    WEIGHT_INT4_VALUES.view(np.uint32).astype(np.uint64)[i >> 4] << 32 \
        for i in range(256)
])
WEIGHT_INT3_8_VALUES = np.array([
    WEIGHT_INT3_VALUES.view(np.uint32).astype(np.uint64)[i % 8] | \
    WEIGHT_INT3_VALUES.view(np.uint32).astype(np.uint64)[i >> 3] << 32 \
        for i in range(64)
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
SCALE_SE5M2_VALUES = np.copysign((np.arange(0,256, dtype=np.uint32)%128+445<<21).view(np.float32),np.arange(127,-129,-1,dtype=np.float32))
SCALE_UE5M1_VALUES = ((np.arange(0,256,dtype=np.uint32)&63)+223<<22).view(np.float32)
# This is used by the 2d plane quantization; include the sqrt(2) normalization inside it
SCALE_UE5M1_VALUES_NORM = SCALE_UE5M1_VALUES * np.sqrt(2)

SUB_SCALES_BINS = 16
SCALE_INT4_VALUES = np.arange(1, 17, dtype=np.float32) / 16
# Same trick as the WEIGHT_INT4_8_VALUES but for the subscales
SCALE_INT4_8_VALUES = np.array([
    SCALE_INT4_VALUES.view(np.uint32).astype(np.uint64)[i % 16] | \
    SCALE_INT4_VALUES.view(np.uint32).astype(np.uint64)[i >> 4] << 32 \
        for i in range(256)
])

@dataclass
class QuantConfig:
    alg: Literal[
        "Q8L", "Q8M", "Q8S", 
        "Q6L", "Q6M", "Q6S", 
        "Q5L", "Q5M", "Q5S", 
        "Q4L", "Q4M", "Q4S", 
        "Q3L", "Q3M", "Q3S"
    ]

    def element_size(self) -> float:
        """element size in bits per weight"""
        return self._BITS() + {"L": 0.5625, "M": 0.3125, "S": 0.1875}[self.alg[2]]
    
    def _BITS(self) -> int:
        return int(self.alg[1])
    
    def _USE_SIGN(self) -> bool:
        return self._BITS() <= 4
    
    def _USE_NONLINEAR(self) -> bool:
        return self._BITS() >= 5 or self.alg == "Q4L"
    
    def _QUANT_SUPER_BLOCK(self) -> int:
        return 128

    def _QUANT_SUB_BLOCK(self) -> int:
        return {"L": 8, "M": 16, "S": 32}[self.alg[2]]
    
    def _SUB_BLOCKS_PER_SUPER(self) -> int:
        return self._QUANT_SUPER_BLOCK() // self._QUANT_SUB_BLOCK()

def unpack_fp8_scales(scales: np.ndarray, config: QuantConfig) -> np.ndarray:
    if config._USE_SIGN():
        return SCALE_SE5M2_VALUES[scales]
    return SCALE_UE5M3_VALUES[scales]

def dequantize_nonlinear(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    
    if type(quants) is not np.ndarray: # Convert to ndarray if needed
        quants = np.array(quants, dtype=np.uint8)
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config.element_size())

    # Unpack scales
    superBlocks = dims // config._QUANT_SUPER_BLOCK()
    subBlocks = dims // config._QUANT_SUB_BLOCK()
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subScales = quants[..., -(subBlocks // 2):]

    superScales = unpack_fp8_scales(superScales, config)
    subScales = SCALE_INT4_8_VALUES[subScales].view(np.float32)
    # Use inplace reshape to merge both scale levels
    subScales = (subScales.reshape((-1, config._SUB_BLOCKS_PER_SUPER())) * superScales.reshape(-1, 1)).reshape(subScales.shape)
    
    # Unpack weights
    eighth = superBlocks * config._QUANT_SUPER_BLOCK() // 8
    if config._BITS() == 8:
        quants = WEIGHT_INT8_VALUES[quants[..., :eighth * 8]]
    elif config._BITS() == 6:
        # Unpack the 6 bit weights into a single buffer, without using temporary arrays
        unpacked = np.empty((*quants.shape[:-1], dims), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 6], 0b11, out=unpacked[..., :eighth * 6])
        unpacked[..., eighth * 2:eighth * 4] <<= 2
        unpacked[..., :eighth * 2] |= unpacked[..., eighth * 2:eighth * 4]
        unpacked[..., eighth * 4:eighth * 6] <<= 4
        unpacked[..., :eighth * 2] |= unpacked[..., eighth * 4:eighth * 6]
        np.right_shift(quants[..., :eighth * 6], 2, out=unpacked[..., eighth * 2:])
        quants = WEIGHT_INT6_VALUES[unpacked]
    elif config._BITS() == 5:
        unpacked = np.unpackbits(quants[..., eighth * 4:eighth * 5], axis=-1)
        unpacked <<= 4
        unpacked[..., :eighth * 4] |= quants[..., :eighth * 4] >> 4
        unpacked[..., eighth * 4:] |= quants[..., :eighth * 4] & 0xF
        quants = WEIGHT_INT5_VALUES[unpacked]
    elif config._BITS() == 4:
        quants = quants[..., :eighth * 4]
        quants = WEIGHT_INT4_8_VALUES[quants].view(np.float32)
    elif config._BITS() == 3:
        # Unpack the 3 bit weights into a single buffer, with 6 bit integers 
        # representing 2 weights, without using temporary arrays
        unpacked = np.empty((*quants.shape[:-1], dims // 2), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 3], 0b11, out=unpacked[..., :eighth * 3])
        unpacked[..., eighth:eighth * 2] <<= 2
        unpacked[..., :eighth] |= unpacked[..., eighth:eighth * 2]
        unpacked[..., eighth * 2:eighth * 3] <<= 4
        unpacked[..., :eighth] |= unpacked[..., eighth * 2:eighth * 3]
        np.right_shift(quants[..., :eighth * 3], 2, out=unpacked[..., eighth:])
        quants = WEIGHT_INT3_8_VALUES[unpacked].view(np.float32)
    elif config._BITS() == 2:
        out = quants[..., :eighth * 2]
        out = np.concatenate([out >> 6, out >> 4 & 3, out >> 2 & 3, out & 3], axis=-1)
        quants = WEIGHT_INT2_VALUES[out]

    # Use inplace reshape to multiply the elements by their respective scale
    reshaped = quants.reshape((-1, config._QUANT_SUB_BLOCK()))
    reshaped *= subScales.reshape((-1, 1))
    return quants

def dequantize_2d(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    
    if type(quants) is not np.ndarray: # Convert to ndarray if needed
        quants = np.array(quants, dtype=np.uint8)
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config.element_size())

    # Unpack scales
    superBlocks = dims // config._QUANT_SUPER_BLOCK()
    subBlocks = dims // config._QUANT_SUB_BLOCK()
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subScales = quants[..., -(subBlocks // 2):]

    spins = superScales & 0xC0
    superScales = SCALE_UE5M1_VALUES_NORM[superScales]
    subScales = SCALE_INT4_8_VALUES[subScales].view(np.float32)
    # Use inplace reshape to merge both scale levels
    subScales = (subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER()) * superScales.reshape(-1, 1)).reshape(subScales.shape)
    
    # Unpack weights
    eighth = superBlocks * config._QUANT_SUPER_BLOCK() // 8
    if config._BITS() == 4:
        tmp = quants[..., :eighth * 4].astype(np.uint16)
        spinMerge = tmp.reshape(-1, config._QUANT_SUPER_BLOCK() // 2)
        spinMerge |= (spins.astype(np.uint16) << 2).reshape(-1, 1)
        quants = WEIGHT_INT4_2D_VALUES[tmp].view(np.float32)

    if config._BITS() == 3:
        # Unpack the 3 bit weights into a single buffer, with 6 bit integers 
        # representing 2 weights, without using temporary arrays
        unpacked = np.empty((*quants.shape[:-1], dims // 2), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 3], 0b11, out=unpacked[..., :eighth * 3])
        unpacked[..., eighth:eighth * 2] <<= 2
        unpacked[..., :eighth] |= unpacked[..., eighth:eighth * 2]
        unpacked[..., eighth * 2:eighth * 3] <<= 4
        unpacked[..., :eighth] |= unpacked[..., eighth * 2:eighth * 3]
        np.right_shift(quants[..., :eighth * 3], 2, out=unpacked[..., eighth:])
        spinMerge = unpacked.reshape(-1, config._QUANT_SUPER_BLOCK() // 2)
        spinMerge |= spins.reshape(-1, 1)
        quants = WEIGHT_INT3_2D_VALUES[unpacked].view(np.float32)
        
    if config._BITS() == 2:
        unpacked = np.empty((*quants.shape[:-1], dims // 2), dtype=np.uint8)
        np.bitwise_and(quants[..., :eighth * 2], 0xF, out=unpacked[..., :unpacked.shape[-1]//2])
        np.right_shift(quants[..., :eighth * 2], 4, out=unpacked[..., unpacked.shape[-1]//2:])
        spinMerge = unpacked.reshape(-1, config._QUANT_SUPER_BLOCK() // 2)
        spinMerge |= spins.reshape(-1, 1)
        quants = WEIGHT_INT2_2D_VALUES[unpacked].view(np.float32)

    # Use inplace reshape to multiply the elements by their respective scale
    quants = (quants.reshape(-1, config._QUANT_SUB_BLOCK()) * subScales.reshape(-1, 1)).reshape(quants.shape)
    return quants

def dequantize(quants: np.ndarray | list, config: QuantConfig) -> np.ndarray:
    """
    Returns a quantized embedding vector back to floats
    Params:
        quants: quantized vector or matrix
        config: configuration used for quantizing the data
    """
    if config._USE_NONLINEAR():
        return dequantize_nonlinear(quants, config)
    return dequantize_2d(quants, config)

def quantize_nonlinear(
        data: list[float] | np.ndarray, config: QuantConfig, optimize: int = OPTIMIZE_FAST, calibration: np.ndarray | None = None
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
    qtypeMin = -(1 << config._BITS() - 1)
    qtypeMax = (1 << config._BITS() - 1) - 1
    numSuperBlocks = len(data) // config._QUANT_SUPER_BLOCK()
    numSubBlocks = len(data) // config._QUANT_SUB_BLOCK()
    superBiases = np.full((numSuperBlocks), 0.0, dtype=np.float32)
    subBiases = np.full((numSubBlocks), {8: 0.45, 6: 0.35, 5: 0.2, 4: 0.0, 3: -0.35, 2: -2.75}[config._BITS()], dtype=np.float32)
    signs = np.full((numSuperBlocks), 0, dtype=np.uint32)
    superBlockErrors = [1e10] * numSuperBlocks

    def pack(data: np.ndarray, bits: int) -> np.ndarray:
        if bits == 8:
            return data
        if bits == 6:
            rest = data[..., :data.shape[-1]//4]
            low = np.concatenate([rest, rest >> 2, rest >> 4], axis=-1) & 0b11
            return (data[..., data.shape[-1]//4:] << 2) | low
        if bits == 5:
            lower = data & 0xF
            upper = np.packbits(data >> 4, axis=-1)
            return np.concatenate([(lower[..., :data.shape[-1]//2] << 4) | lower[..., data.shape[-1]//2:], upper], axis=-1)
        if bits == 4:
            return data[..., 0::2] | (data[..., 1::2] << 4)
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
        quants = quants.astype(np.uint8) + (1 << config._BITS() - 1)
        superScales = (superScales).astype(np.uint8)
        subScales = (subScales - 1).astype(np.uint8)
        return np.concatenate([pack(quants, config._BITS()), superScales, pack(subScales, 4)], axis=-1)

    # For testing biases when optimizing quantization
    def dequantize_internal(
            quants: np.ndarray, superScales: np.ndarray, subScales: np.ndarray,
        ) -> np.ndarray:
        # Unpack weights
        quants *= 1 / ((1 << config._BITS() - 1) - 1)
        quants = np.abs(quants) * quants + quants + quants  # x^2 + 2x keeping sign
        # Unpack scales
        superScales = unpack_fp8_scales(superScales, config)
        superScales *= (0.33333 / SUB_SCALES_BINS)  # Multipliers for subscales and weights
        # Use inplace reshape to merge both scale levels
        subScales = (subScales.reshape((-1, config._SUB_BLOCKS_PER_SUPER())) * superScales.reshape(-1, 1)).reshape(subScales.shape)
        # Use inplace reshape to multiply the elements by their respective scale
        quants = (quants.reshape((-1, config._QUANT_SUB_BLOCK())) * subScales.reshape(-1, 1)).reshape(quants.shape)
        return quants
    
    def quantize_internal_step(
            data: np.ndarray, superBias: np.ndarray, subBias: np.ndarray, tiling: int = 1
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Constants
        e = 5
        m = (8 - config._USE_SIGN()) - e
        # Scales are the max absolute value of each block. If we are using tiling
        # to test multiples biases for optimization we know that the maximums will 
        # also be tiled.
        absolutes = np.abs(data[:len(data)//tiling])
        subMaximums = absolutes.reshape((-1, config._QUANT_SUB_BLOCK()), copy=False)
        subMaximums = np.max(subMaximums, axis=1)
        superMaximums = absolutes.reshape((-1, config._QUANT_SUPER_BLOCK()), copy=False)
        superMaximums = np.max(superMaximums, axis=1)
        if tiling != 1:  # Tile the maximums
            subMaximums = np.tile(subMaximums, tiling)
            superMaximums = np.tile(superMaximums, tiling)
        
        # Extract mantissa and exponent 
        mantissa, exponent = np.frexp(superMaximums)
        mantissa = np.ceil((mantissa - (0.5 + 0.5 / (1<<m))) * (1<<m+1)).astype(np.int32)
        exponent = np.maximum(exponent + (1<<e-1)-1, 0)
        # Pack both into an uint8.
        superScales = np.clip((exponent << m) + mantissa + superBias.astype(np.int32), 0, (1<<e+m)-1)
        # Dequantize scales and normalize maximums for the second level of scales
        unpackedScales = unpack_fp8_scales(superScales, config)

        # Second level quantization
        subMaximums = (subMaximums.reshape((-1, config._SUB_BLOCKS_PER_SUPER())) / unpackedScales.reshape(-1, 1)).reshape(subMaximums.shape)
        subScales = np.clip(np.round(subMaximums * SUB_SCALES_BINS + subBias), 1, SUB_SCALES_BINS)

        # Mix both scales for weight normalization
        outputShape = subScales.shape
        dqScales = subScales.reshape((-1, config._SUB_BLOCKS_PER_SUPER()), copy=False)
        dqScales = dqScales * (unpackedScales.reshape(-1, 1, copy=False) / np.float32(SUB_SCALES_BINS))
        dqScales = dqScales.reshape(outputShape, copy=False)

        # Normalize weights for later quantization. Use inplace reshape to avoid an np.repeat().
        outputShape = data.shape
        quants = data.reshape((-1, config._QUANT_SUB_BLOCK()), copy=False)
        quants = quants / dqScales.reshape(-1, 1, copy=False)
        quants = quants.reshape(outputShape, copy=False)
        
        squaredMax = qtypeMax * qtypeMax
        # Final weights are quantized using a non linear function
        #quants = np.copysign(np.sqrt(np.abs(quants) * 3 + 1) - 1, quants) * dtypeMax
        quants = np.round(np.copysign(np.sqrt(np.abs(quants) * (3 * squaredMax) + squaredMax) - qtypeMax, quants))
        return quants, superScales, subScales
    
    def quantize_internal_final(
            quants: np.ndarray, superScales: np.ndarray, signs: np.ndarray | None
        ) -> tuple[np.ndarray, np.ndarray]:
        # Final step of quantization; apply signs and clip quantized weights
        if signs is not None:
            superScales |= signs << 7
            casted = quants.view(np.uint32).reshape((-1, config._QUANT_SUPER_BLOCK()), copy=False)
            casted ^= (signs << 31).reshape(-1, 1, copy=False)
        quants = np.clip(quants, qtypeMin, qtypeMax)
        return quants, superScales

    # Doing a round() for the super scales is not always optimal. Try
    # to find a better quantization by testing different biases for it.
    if optimize != OPTIMIZE_FAST:

        superBiasRange = [
            [None, None, (  1, 3), ( 0, 2), ( 0, 2), ( 0, 2), ( 0, 2), None, ( 0, 2)],
            [None, None, (-12,12), (-2, 2), (-2, 3), (-1, 4), (-1, 4), None, ( 0, 5)],
        ][optimize][config._BITS()]
        subBiasRange = [
            [None, None, ( -4, 1), (-2, 1), (-2, 1), ( 0, 2), ( 0, 2), None, ( 0, 2)],
            [None, None, (-16,16), (-7, 6), (-3, 5), (-1, 8), (-1, 8), None, ( 0,10)],
        ][optimize][config._BITS()]
        
        superBiasLength = superBiasRange[1] - superBiasRange[0]
        subBiasLength = subBiasRange[1] - subBiasRange[0]

        tiledData = np.tile(data, subBiasLength * superBiasLength)
        if calibration is not None:
            tiledCalibration = np.tile(calibration, subBiasLength * superBiasLength)
        testSuperBias = np.repeat(np.arange(superBiasRange[0], superBiasRange[1], 1, dtype=np.float32), subBiasLength * numSuperBlocks)
        testSubBias = np.tile(np.repeat(np.arange(subBiasRange[0], subBiasRange[1], 1, dtype=np.float32), numSubBlocks), superBiasLength)

        # Common computations when testing signs
        quants, superScales, subScales = quantize_internal_step(
            data = tiledData,
            superBias = testSuperBias, 
            subBias = testSubBias,
            tiling = subBiasLength * superBiasLength
        )

        for sign in range(0, 1 + config._USE_SIGN()):

            # Now finish the quantization with the computations that require the sign
            if sign == 0:
                dequantized = dequantize_internal(*quantize_internal_final(quants, superScales, None), subScales)
            else:
                dequantized = dequantize_internal(
                    *quantize_internal_final(
                        quants, superScales, np.full((numSuperBlocks * subBiasLength * superBiasLength), sign, dtype=np.uint32)
                    ), subScales
                )

            mse = np.square(tiledData - dequantized)
            if calibration is not None:
                mse *= tiledCalibration
            mse = np.reshape(mse, (-1, config._QUANT_SUB_BLOCK())).mean(axis=1)

            # Find optimal block roundings
            # Test all subbiases at the same to reduce computation time
            mse = mse.reshape((superBiasLength, subBiasLength, numSubBlocks))
            # Find which bias had the lowest error for each subblock in each superblock
            optimalSubBiases = np.argmin(mse, axis=1) + subBiasRange[0]
            # With the optimal subblock rounding found, calculate the error for each superblock bias
            biasMse = np.min(mse, axis=1).reshape((superBiasLength, numSuperBlocks, -1)).mean(axis=2)

            for biasIdx in range(superBiasLength):
                # Check errors per superblock
                for block in range(numSuperBlocks):
                    if biasMse[biasIdx][block] < superBlockErrors[block]:
                        # If the error with this superbias is lower overwrite
                        superBiases[block] = biasIdx + superBiasRange[0]
                        superBlockErrors[block] = biasMse[biasIdx][block]
                        signs[block] = sign
                        # Overwrite subbiases
                        start = block * config._SUB_BLOCKS_PER_SUPER()
                        end = (block + 1) * config._SUB_BLOCKS_PER_SUPER()
                        subBiases[start:end] = optimalSubBiases[biasIdx][start:end]

    quants, superScales, subScales = quantize_internal_step(data, superBiases, subBiases)
    quants, superScales = quantize_internal_final(
        quants, superScales, signs if config._USE_SIGN() and optimize != OPTIMIZE_FAST else None
    )

    # Combine all components of quantization and 
    # reshape to the expected output shape
    inputShape[-1] = -1
    return combine(
        quants.reshape(inputShape), 
        superScales.reshape(inputShape),
        subScales.reshape(inputShape)
    )
    #return dequantize_combine(quants, superScales, subScales, config)

def quantize_2d(
        data: list[float] | np.ndarray, config: QuantConfig, optimize: int = OPTIMIZE_FAST, calibration: np.ndarray | None = None
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

    # Separate values into x and y coordinates
    axis0 = data[0::2]
    axis1 = data[1::2]
    # Convert the 2d vectors into magnitude and angle. Normalize the magnitudes, 
    # so that the maximum representable range stays as a power of 2.
    magnitudes = np.sqrt(np.square(axis0) + np.square(axis1)) / np.sqrt(2)
    angles = np.arctan2(axis1, axis0) % (2*np.pi)

    # Other variables initialization
    bitValues = 1 << config._BITS()
    numSuperBlocks = len(magnitudes) // (config._QUANT_SUPER_BLOCK() // 2)
    numSubBlocks = len(magnitudes) // (config._QUANT_SUB_BLOCK() // 2)
    superBiases = np.full((numSuperBlocks), 0.5)
    subBiases = np.full((numSubBlocks), 0.5)
    bestSpins = np.full((numSuperBlocks), 0.0)
    superBlockErrors = [1e10] * numSuperBlocks

    def pack(data: np.ndarray, bits: int) -> np.ndarray:
        if bits == 8:
            return data
        if bits == 6:
            rest = data[..., :data.shape[-1]//4]
            low = np.concatenate([rest, rest >> 2, rest >> 4], axis=-1) & 0b11
            return (data[..., data.shape[-1]//4:] << 2) | low
        if bits == 4:
            return data[..., :data.shape[-1]//2] | (data[..., data.shape[-1]//2:] << 4)
        
    def pack_subscales(scales: np.ndarray) -> np.ndarray:
        return scales[..., 0::2] | (scales[..., 1::2] << 4)
        
    def combine(
            magnitudes: np.ndarray, angles: np.ndarray, superScales: np.ndarray, subScales: np.ndarray, spins: np.ndarray,
        ) -> np.ndarray:
        # Pack quantized magnitudes and angles into the final dtype
        spins = (spins * (4 * 64)).astype(np.uint8)
        quants = np.maximum(magnitudes * (bitValues + 1) + angles - bitValues, 0).astype(np.uint8)
        subScales = (subScales - 1).astype(np.uint8)
        # Fit everything into their respective uint types
        return np.concatenate([pack(quants, config._BITS() * 2), superScales | spins, pack_subscales(subScales)], axis=-1).astype(np.uint8)
    
    # For testing biases when optimizing quantization
    def dequantize_internal(
            magnitudes: np.ndarray, angles: np.ndarray, superScales: np.ndarray, subScales: np.ndarray, spin: float
        ) -> np.ndarray:

        # Unpack scales and combine them
        superScales = SCALE_UE5M1_VALUES[superScales]
        superScales *= np.sqrt(2) / 16  # Multipliers for subscales and weights
        subScales = (subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER()) * superScales.reshape(-1, 1)).reshape(subScales.shape)

        # Merge magnitudes, angles and spins, and combine with the scales
        values = np.maximum(magnitudes * (bitValues + 1) + angles - bitValues, 0)
        if config._BITS() == 4:
            result = WEIGHT_INT4_2D_VALUES[values.astype(np.uint16) | (np.uint16(spin * 4) << 8)].view(np.float32)
        elif config._BITS() == 3:
            result = WEIGHT_INT3_2D_VALUES[values.astype(np.uint8) | (np.uint8(spin * 4) << 6)].view(np.float32)
        elif config._BITS() == 2:
            result = WEIGHT_INT2_2D_VALUES[values.astype(np.uint8) | (np.uint8(spin * 4) << 6)].view(np.float32)
        result = (result.reshape(-1, config._QUANT_SUB_BLOCK()) * subScales.reshape(-1, 1)).reshape(result.shape)
        return result

    def quantize_internal_step(
            magnitudes: np.ndarray, angles: np.ndarray, superBias: np.ndarray, subBias: np.ndarray, tiling: int = 1
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Constants
        e = 5
        m = 6 - e
        # Scales are the max absolute value of each block. If we are using tiling
        # to test multiples biases for optimization we know that the maximums will 
        # also be tiled.
        absolutes = np.abs(magnitudes[:len(magnitudes)//tiling])
        subMaximums = absolutes.reshape(-1, config._QUANT_SUB_BLOCK() // 2, copy=False)
        subMaximums = np.max(subMaximums, axis=1)
        superMaximums = absolutes.reshape(-1, config._QUANT_SUPER_BLOCK() // 2, copy=False)
        superMaximums = np.max(superMaximums, axis=1)
        if tiling != 1:  # Tile the maximums
            subMaximums = np.tile(subMaximums, tiling)
            superMaximums = np.tile(superMaximums, tiling)
        
        # Extract mantissa and exponent 
        mantissa, exponent = np.frexp(superMaximums)
        mantissa = np.ceil((mantissa - (0.5 + 0.5 / (1<<m))) * (1<<m+1)).astype(np.int32)
        exponent = np.maximum(exponent + (1<<e-1)-1, 0)
        # Pack both into an uint8.
        superScales = np.clip((exponent << m) + mantissa + superBias.astype(np.int32), 0, (1<<e+m)-1)
        # Dequantize scales and normalize maximums for the second level of scales
        unpackedScales = SCALE_UE5M1_VALUES[superScales]

        # Second level quantization
        subMaximums = (subMaximums.reshape(-1, config._SUB_BLOCKS_PER_SUPER()) / unpackedScales.reshape(-1, 1)).reshape(subMaximums.shape)
        subScales = np.clip(np.round(subMaximums * SUB_SCALES_BINS + subBias), 1, SUB_SCALES_BINS)

        # Mix both scales for weight normalization
        outputShape = subScales.shape
        dqScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER(), copy=False)
        dqScales = dqScales * (unpackedScales.reshape(-1, 1, copy=False) / np.float32(SUB_SCALES_BINS))
        dqScales = dqScales.reshape(outputShape, copy=False)

        # Normalize weights for later quantization. Use inplace reshape to avoid an np.repeat().
        outputShape = magnitudes.shape
        magnitudes = magnitudes.reshape(-1, config._QUANT_SUB_BLOCK() // 2, copy=True)
        magnitudes /= dqScales.reshape(-1, 1, copy=False)
        magnitudes = magnitudes.reshape(outputShape, copy=False)

        # Reference axis to find the closest point in the codebook
        reference = (magnitudes * np.cos(angles), magnitudes * np.sin(angles))
        # Quantize the magnitudes and prepare the angles
        magnitudes = np.clip(np.ceil(magnitudes * (bitValues - 1)), 1, bitValues - 1)
        angles = angles * ((bitValues + 1) / (2 * np.pi))

        return magnitudes, angles, superScales, subScales, reference
    
    def quantize_internal_final(
            magnitudes: np.ndarray, angles: np.ndarray, spins: np.ndarray, reference: tuple[np.ndarray, np.ndarray], calibration: np.ndarray | None,
        ) -> tuple[np.ndarray, np.ndarray]:

        anglesStep = (bitValues + 1) / (2 * np.pi)
        anglesShift0 = (magnitudes.reshape(-1, config._QUANT_SUPER_BLOCK() // 2) * 0.5 + spins.reshape(-1, 1)).reshape(magnitudes.shape)
        anglesShift1 = anglesShift0 - 0.5
        angles0 = np.round(angles - anglesShift0)
        angles1 = np.round(angles - anglesShift1)

        dqAngles0 = (angles0 + anglesShift0) / anglesStep
        dqAngles1 = (angles1 + anglesShift1) / anglesStep
        dqMagnitudes0 = magnitudes / (bitValues - 1)
        dqMagnitudes1 = dqMagnitudes0 - (1 / (bitValues - 1))

        if calibration is None:
            error0 = np.square(dqMagnitudes0 * np.cos(dqAngles0) - reference[0]) + \
                np.square(dqMagnitudes0 * np.sin(dqAngles0) - reference[1])
            error1 = np.square(dqMagnitudes1 * np.cos(dqAngles1) - reference[0]) + \
                np.square(dqMagnitudes1 * np.sin(dqAngles1) - reference[1])
        else:
            a = calibration[0::2]
            b = calibration[1::2]
            error0 = a * np.square(dqMagnitudes0 * np.cos(dqAngles0) - reference[0]) + \
                b * np.square(dqMagnitudes0 * np.sin(dqAngles0) - reference[1])
            error1 = a * np.square(dqMagnitudes1 * np.cos(dqAngles1) - reference[0]) + \
                b * np.square(dqMagnitudes1 * np.sin(dqAngles1) - reference[1])
        
        selector = (error1 <= error0)
        outMagnitudes = magnitudes - selector
        outAngles = np.where(selector, angles1, angles0)
        outAngles = np.fmod(outAngles + (bitValues + 1), bitValues + 1)
        return outMagnitudes, outAngles
    
    # Doing a round() for the scales is not always optimal. Try
    # to find a better quantization by testing different biases for it.
    if optimize != OPTIMIZE_FAST:

        superBiasRange = [
            [None, None, ( 0, 1), ( 0, 1), ( 0, 1)],
            [None, None, (-1, 1), ( 0, 2), ( 0, 2)],
        ][optimize][config._BITS()]
        subBiasRange = [
            [None, None, (-2, 2), ( 0, 4), ( 0, 4)],
            [None, None, (-4, 6), (-1, 8), (-1, 8)],
        ][optimize][config._BITS()]

        superBiasLength = superBiasRange[1] - superBiasRange[0]
        subBiasLength = subBiasRange[1] - subBiasRange[0]

        tiledData = np.tile(data, subBiasLength * superBiasLength)
        if calibration is not None:
            tiledCalibration = np.tile(calibration, subBiasLength * superBiasLength)
        tiledMagnitudes = np.tile(magnitudes, subBiasLength * superBiasLength)
        tiledAngles = np.tile(angles, subBiasLength * superBiasLength)

        normalizedMagnitudes, normalizedAngles, superScales, subScales, reference = quantize_internal_step(
            magnitudes = tiledMagnitudes, 
            angles = tiledAngles,
            superBias = np.repeat(np.arange(superBiasRange[0], superBiasRange[1], 1), subBiasLength * numSuperBlocks), 
            subBias = np.tile(np.repeat(np.arange(subBiasRange[0], subBiasRange[1], 1), numSubBlocks), superBiasLength),
            tiling = subBiasLength * superBiasLength,
        )

        for spin in [0, 0.25, 0.5, 0.75]:

            spinArray = np.full((numSuperBlocks * superBiasLength * subBiasLength), spin)
            dequantized = dequantize_internal(
                *quantize_internal_final(
                    magnitudes = normalizedMagnitudes,
                    angles = normalizedAngles,
                    spins = spinArray,
                    reference = reference,
                    calibration = None if calibration is None else tiledCalibration,
                ),
                superScales, subScales, spin
            )

            mse = np.square(tiledData - dequantized)
            if calibration is not None:
                mse *= tiledCalibration
            mse = np.reshape(mse, (-1, config._QUANT_SUB_BLOCK())).mean(axis=1)

            # Find optimal block roundings
            # Test all subbiases at the same to reduce computation time
            mse = mse.reshape((superBiasLength, subBiasLength, numSubBlocks))
            # Find which bias had the lowest error for each subblock in each superblock
            optimalSubBiases = np.argmin(mse, axis=1) + subBiasRange[0]
            # With the optimal subblock rounding found, calculate the error for each superblock bias
            biasMse = np.min(mse, axis=1).reshape((superBiasLength, numSuperBlocks, -1)).mean(axis=2)

            for biasIdx in range(superBiasLength):
                # Check errors per superblock
                for block in range(numSuperBlocks):
                    if biasMse[biasIdx][block] < superBlockErrors[block]:
                        # If the error with this superbias is lower overwrite
                        superBiases[block] = biasIdx + superBiasRange[0] 
                        superBlockErrors[block] = biasMse[biasIdx][block]
                        # Also overwrite subbiases
                        start = block * config._SUB_BLOCKS_PER_SUPER()
                        end = (block + 1) * config._SUB_BLOCKS_PER_SUPER()
                        subBiases[start:end] = optimalSubBiases[biasIdx][start:end]
                        bestSpins[block] = spin
    
    # Generate final quantized vector
    normalizedMagnitudes, normalizedAngles, superScales, subScales, reference = quantize_internal_step(
        magnitudes, angles, superBiases, subBiases, 1
    )
    outMagnitudes, outAngles = quantize_internal_final(normalizedMagnitudes, normalizedAngles, bestSpins, reference, calibration)

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
        data: list[float] | np.ndarray, config: QuantConfig, optimize: int = OPTIMIZE_FAST, calibration: np.ndarray | None = None
    ) -> np.ndarray:
    if config._USE_NONLINEAR():
        return quantize_nonlinear(data, config, optimize, calibration)
    return quantize_2d(data, config, optimize, calibration)