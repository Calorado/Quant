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

from quantization import *
import cupy as cu

CU_FP16_WEIGHT_INT8_VALUES = cu.array(WEIGHT_INT8_VALUES, dtype=np.float16)
"""CU_FP16_WEIGHT_INT8_8_VALUES = cu.array([
    WEIGHT_INT8_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 256] | \
    WEIGHT_INT8_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i >> 8] << 16 \
        for i in range(65536)
], dtype=cu.uint32)"""
CU_FP16_WEIGHT_INT6_8_VALUES = cu.array([
    WEIGHT_INT6_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 64] | \
    WEIGHT_INT6_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i >> 6] << 16 \
        for i in range(4096)
], dtype=cu.uint32)
CU_FP16_WEIGHT_INT5_8_VALUES = cu.array([
    WEIGHT_INT5_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 32] | \
    WEIGHT_INT5_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i >> 5] << 16 \
        for i in range(1024)
], dtype=cu.uint32)
CU_FP16_WEIGHT_INT4_8_VALUES = cu.array([
    WEIGHT_INT4_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 16] | \
    WEIGHT_INT4_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i >> 4] << 16 \
        for i in range(256)
], dtype=cu.uint32)

CU_FP16_WEIGHT_INT4_2D_VALUES = cu.empty((1024), dtype=cu.uint32)
for i in range(1024):
    magnitude, angle, spin = ((i % 256 + 16) // 17, (i % 256 + 16) % 17, i >> 8)
    angle = (angle + magnitude / 2 + spin / 4) / (17 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 15).astype(np.float16).view(np.uint16).astype(np.uint32)
    b = (np.sin(angle) * magnitude / 15).astype(np.float16).view(np.uint16).astype(np.uint32)
    CU_FP16_WEIGHT_INT4_2D_VALUES[i] = a | (b << 16)

CU_FP16_WEIGHT_INT3_2D_VALUES = cu.empty((256), dtype=cu.uint32)
for i in range(256):
    magnitude, angle, spin = ((i % 64 + 8) // 9, (i % 64 + 8) % 9, i >> 6)
    angle = (angle + magnitude / 2 + spin / 4) / (9 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 7).astype(np.float16).view(np.uint16).astype(np.uint32)
    b = (np.sin(angle) * magnitude / 7).astype(np.float16).view(np.uint16).astype(np.uint32)
    CU_FP16_WEIGHT_INT3_2D_VALUES[i] = a | (b << 16)

CU_FP16_WEIGHT_INT2_2D_VALUES = cu.empty((256), dtype=cu.uint32)
for i in range(256):
    magnitude, angle, spin = ((i % 16 + 4) // 5, (i % 16 + 4) % 5, i >> 6)
    angle = (angle + magnitude / 2 + spin / 4) / (5 / (2*np.pi))
    a = (np.cos(angle) * magnitude / 3).astype(np.float16).view(np.uint16).astype(np.uint32)
    b = (np.sin(angle) * magnitude / 3).astype(np.float16).view(np.uint16).astype(np.uint32)
    CU_FP16_WEIGHT_INT2_2D_VALUES[i] = a | (b << 16)

CU_FP16_SCALE_INT4_8_VALUES = cu.array([
    SCALE_INT4_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 16] | \
    SCALE_INT4_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i >> 4] << 16 \
        for i in range(256)
], dtype=cu.uint32)

CU_FP16_SCALE_INT2_8_VALUES = cu.array([
    SCALE_INT2_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[i % 4] | \
    SCALE_INT2_VALUES.astype(np.float16).view(np.uint16).astype(np.uint32)[(i >> 2) % 4] << 16 \
        for i in range(256)
], dtype=cu.uint32)

CU_FP16_SCALE_UE5M3_VALUES = cu.array(SCALE_UE5M3_VALUES, dtype=cu.float16)

def cu_dequantize_nonlinear(quants: cu.ndarray, config: QuantConfig, out: cu.ndarray | None = None) -> cu.ndarray:
    """
    Returns a quantized embedding vector back to floats
    Params:
        quants: quantized vector or matrix
        config: configuration used for quantizing the data
    """
    
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config.element_size())
    superBlocks = dims // config._QUANT_SUPER_BLOCK()
    subBlocks = dims // config._QUANT_SUB_BLOCK()
    if out is None:
        out = cu.empty((*quants.shape[:-1], dims), dtype=cu.float16)

    # Unpack weights
    eighth = dims // 8
    if config._BITS() == 8:
        cu.take(CU_FP16_WEIGHT_INT8_VALUES, quants[..., :eighth * 8], out=out)
    elif config._BITS() == 6:
        scratch = cu.empty(quants[..., eighth * 4:eighth * 6].shape, dtype=cu.uint8)
        unpacked = quants[..., :eighth * 4].astype(cu.uint16)
        unpacked <<= 4
        unpacked[..., :eighth * 2] |= cu.bitwise_and(quants[..., eighth * 4:eighth * 6], 0xF, out=scratch)
        unpacked[..., eighth * 2:] |= cu.right_shift(quants[..., eighth * 4:eighth * 6], 4, out=scratch)
        cu.take(CU_FP16_WEIGHT_INT6_8_VALUES, unpacked, out=out.view(cu.uint32))
    elif config._BITS() == 5:
        unpacked = quants[..., :eighth * 4].astype(cu.uint16)
        unpacked <<= 2
        scratch = cu.empty(quants[..., :eighth].shape, dtype=cu.uint8)
        rest = quants[..., eighth * 4:eighth * 5]
        unpacked[..., :eighth * 1] |= cu.right_shift(rest, 6, out=scratch)
        unpacked[..., eighth * 1:eighth * 2] |= \
            cu.bitwise_and(cu.right_shift(rest, 4, out=scratch), 3, out=scratch)
        unpacked[..., eighth * 2:eighth * 3] |= \
            cu.bitwise_and(cu.right_shift(rest, 2, out=scratch), 3, out=scratch)
        unpacked[..., eighth * 3:] |= cu.bitwise_and(rest, 3, out=scratch)
        cu.take(CU_FP16_WEIGHT_INT5_8_VALUES, unpacked, out=out.view(cu.uint32))
    elif config._BITS() == 4:
        cu.take(CU_FP16_WEIGHT_INT4_8_VALUES, quants[..., :eighth * 4], out=out.view(cu.uint32))
    
    # Unpack scales
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subScales = quants[..., -(subBlocks // 2):]
    superScales = CU_FP16_SCALE_UE5M3_VALUES[superScales]
    subScales = CU_FP16_SCALE_INT4_8_VALUES[subScales].view(cu.float16)
    # Use inplace reshape to merge both scale levels
    reshaped = subScales.reshape((-1, config._SUB_BLOCKS_PER_SUPER()))
    reshaped *= superScales.reshape((-1, 1))

    # Use inplace reshape to multiply the elements by their respective scale
    reshaped = out.reshape((-1, config._QUANT_SUB_BLOCK()))
    reshaped *= subScales.reshape((-1, 1))
    return out

def cu_dequantize_2d(quants: cu.ndarray, config: QuantConfig, out: cu.ndarray | None = None) -> cu.ndarray:
    
    # Number of elements in the first dimension
    dims = int(quants.shape[-1] * 8 / config.element_size())
    superBlocks = dims // config._QUANT_SUPER_BLOCK()
    subBlocks = dims // config._QUANT_SUB_BLOCK()
    superScales = quants[..., -(subBlocks // 2 + superBlocks):-(subBlocks // 2)]
    subInfo = quants[..., -(subBlocks // 2):]
    
    # Unpack spins
    spins = cu.empty((*quants.shape[:-1], subBlocks), dtype=np.uint8 if config._BITS() < 4 else np.uint16)
    cu.left_shift(subInfo, 2, out=spins[..., :spins.shape[-1] // 2])
    spins[..., spins.shape[-1] // 2:] = subInfo
    spins &= 0xC0

    # Unpack weights
    eighth = dims // 8
    if out is None:
        out = cu.empty((*quants.shape[:-1], dims), dtype=cu.float16)
        
    if config._BITS() == 4:
        spins <<= 2
        tmp = quants[..., :eighth * 4].astype(cu.uint16)
        spinMerge = tmp.reshape(-1, config._QUANT_SUB_BLOCK() // 2)
        spinMerge |= spins.reshape(-1, 1)
        cu.take(CU_FP16_WEIGHT_INT4_2D_VALUES, tmp, out=out.view(cu.uint32))

    elif config._BITS() == 3:
        # Unpack the 3 bit weights into a single buffer, with 6 bit integers 
        # representing 2 weights, without using temporary arrays
        unpacked = cu.empty((*quants.shape[:-1], dims // 2), dtype=cu.uint8)
        cu.bitwise_and(quants[..., :eighth * 3], 0b11, out=unpacked[..., :eighth * 3])
        unpacked[..., eighth:eighth * 2] <<= 2
        unpacked[..., :eighth] |= unpacked[..., eighth:eighth * 2]
        unpacked[..., eighth * 2:eighth * 3] <<= 4
        unpacked[..., :eighth] |= unpacked[..., eighth * 2:eighth * 3]
        cu.right_shift(quants[..., :eighth * 3], 2, out=unpacked[..., eighth:])
        spinMerge = unpacked.reshape(-1, config._QUANT_SUB_BLOCK() // 2)
        spinMerge |= spins.reshape(-1, 1)
        cu.take(CU_FP16_WEIGHT_INT3_2D_VALUES, unpacked, out=out.view(cu.uint32))

    elif config._BITS() == 2:
        unpacked = cu.empty((*quants.shape[:-1], dims // 2), dtype=cu.uint8)
        cu.bitwise_and(quants[..., :eighth * 2], 0xF, out=unpacked[..., :unpacked.shape[-1]//2])
        cu.right_shift(quants[..., :eighth * 2], 4, out=unpacked[..., unpacked.shape[-1]//2:])
        spinMerge = unpacked.reshape(-1, config._QUANT_SUB_BLOCK() // 2)
        spinMerge |= spins.reshape(-1, 1)
        cu.take(CU_FP16_WEIGHT_INT2_2D_VALUES, unpacked, out=out.view(cu.uint32))

    # Unpack scales and merge them
    superScales = CU_FP16_SCALE_UE5M3_VALUES[superScales]
    subScales = CU_FP16_SCALE_INT2_8_VALUES[subInfo].view(cu.float16)
    mergedScales = subScales.reshape(-1, config._SUB_BLOCKS_PER_SUPER())
    mergedScales *= superScales.reshape(-1, 1)

    # Use inplace reshape to multiply the elements by their respective scale
    reshaped = out.reshape(-1, config._QUANT_SUB_BLOCK())
    reshaped *= mergedScales.reshape(-1, 1)
    return out

def cu_dequantize(quants: cu.ndarray, config: QuantConfig, out: cu.ndarray | None = None) -> cu.ndarray:
    """
    Returns a quantized embedding vector back to floats
    Params:
        quants: quantized vector or matrix
        config: configuration used for quantizing the data
    """
    if config._USE_NONLINEAR():
        return cu_dequantize_nonlinear(quants, config, out)
    return cu_dequantize_2d(quants, config, out)