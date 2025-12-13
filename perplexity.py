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

from model import QuantizedEmbedding, QuantizedLinear, CalibrationLinear, OPTIMIZE_FAST, OPTIMIZE_STANDARD, OPTIMIZE_THOROUGH

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, HqqConfig
from transformers import Qwen2ForCausalLM
import torch
from tqdm import tqdm

import time
import os

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GGUF_FILE = None #"Qwen2.5-1.5B-Instruct-Q8_0_imat.gguf"
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="fp4", 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
QUANT_CONFIG = {
    "embed": "Q4L",
    "attention": {
        "v": "Q4L",
        "k": "Q4L",
        "q": "Q4L",
        "o": "Q4L",
    },
    "ffn": {
        "up": "Q4L",
        "down": "Q4L",
        "gate": "Q4L",
    },
    "output": "Q4L",
    "optimize": OPTIMIZE_STANDARD
}

def calculate_calibration_matrix(calibration, model):
    """Runs the model on the given calibration data to obtain calibration statistics for quantization"""
    max_length = 512  # Max length of each generation test
    stride = 128  # Sliding window stride
    seq_len = calibration.input_ids.size(1)
    prev_end_loc = 0

    print("\nCalculating calibration matrix...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = calibration.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            model(input_ids, labels=target_ids, return_dict=True)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

def load_quantized_model(model: str, file: str | None, device: str, quant_config: BitsAndBytesConfig | HqqConfig | dict, calibration):
    if file is not None:
        return AutoModelForCausalLM.from_pretrained(
            os.path.dirname(file),
            gguf_file=file,
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    if type(quant_config) is not dict:
        return AutoModelForCausalLM.from_pretrained(
            model, 
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            quantization_config=quant_config,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if type(model) is Qwen2ForCausalLM:
       
        usesTiedEmbeddings = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
        model.graph_buffer = torch.empty(model.model.layers[0].mlp.gate_proj.weight.nelement(), device="cuda", dtype=torch.float16)
        model.model.embed_tokens = QuantizedEmbedding(model.model.embed_tokens, quant=quant_config["embed"], optimize=quant_config["optimize"])
        
        for layer in model.model.layers:
            layer.mlp.gate_proj = CalibrationLinear(layer.mlp.gate_proj)
            layer.mlp.up_proj = CalibrationLinear(layer.mlp.up_proj)
            layer.mlp.down_proj = CalibrationLinear(layer.mlp.down_proj)
            layer.self_attn.q_proj = CalibrationLinear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = CalibrationLinear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = CalibrationLinear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = CalibrationLinear(layer.self_attn.o_proj)
        if usesTiedEmbeddings:
            model.lm_head = QuantizedLinear(model.model.embed_tokens, batches=32, buffer=model.graph_buffer) # The small qwen2 models use tied embeddings
        else:
            model.lm_head = CalibrationLinear(model.lm_head)

        if quant_config["optimize"] != OPTIMIZE_FAST:
            calculate_calibration_matrix(calibration, model)

        for layer in model.model.layers:
            layer.mlp.gate_proj = QuantizedLinear(layer.mlp.gate_proj, quant=quant_config["ffn"]["gate"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.mlp.up_proj = QuantizedLinear(layer.mlp.up_proj, quant=quant_config["ffn"]["up"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.mlp.down_proj = QuantizedLinear(layer.mlp.down_proj, quant=quant_config["ffn"]["down"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.self_attn.q_proj = QuantizedLinear(layer.self_attn.q_proj, quant=quant_config["attention"]["q"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.self_attn.k_proj = QuantizedLinear(layer.self_attn.k_proj, quant=quant_config["attention"]["k"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.self_attn.v_proj = QuantizedLinear(layer.self_attn.v_proj, quant=quant_config["attention"]["v"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
            layer.self_attn.o_proj = QuantizedLinear(layer.self_attn.o_proj, quant=quant_config["attention"]["o"], optimize=quant_config["optimize"], buffer=model.graph_buffer)
        if not usesTiedEmbeddings:
            model.lm_head = QuantizedLinear(model.lm_head, quant=quant_config["output"], optimize=quant_config["optimize"], batches=32, buffer=model.graph_buffer)

    return model

def get_memory_footprint(model, quant_config = None) -> int:
    if type(quant_config) is dict:
        size = 0
        unique_weights = set() # We might have tied embeddings
        for name, module in model.named_modules():
            if type(module) is not QuantizedEmbedding and \
                type(module) is not QuantizedLinear:
                continue
            if id(module.quantizedWeights) in unique_weights:
                continue
            size += int(module.nelement() * module.element_size())
            unique_weights.add(id(module.quantizedWeights))
        return size
    return model.get_memory_footprint()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
text = open("validation.txt", "r", encoding="utf-8").read()
validation = tokenizer(text, return_tensors="pt")
text = open("calibration.txt", "r", encoding="utf-8").read()
calibration = tokenizer(text, return_tensors="pt")

ts = time.perf_counter()
quantized_model = load_quantized_model(
    MODEL_ID, GGUF_FILE, DEVICE, QUANT_CONFIG, calibration
)
te = time.perf_counter()

unquantized_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

print(f"Quantization time: {te - ts}")
print(f"Original model size: {get_memory_footprint(unquantized_model)}")
print(f"Quantized model size: {get_memory_footprint(quantized_model, QUANT_CONFIG)}")

max_length = 512  # Max length of each generation test
stride = 128  # Sliding window stride
seq_len = validation.input_ids.size(1)

nll_sum = 0.0
kld_sum = 0.0
n_tokens = 0
n_runs = 0
prev_end_loc = 0
inference_time = 0  # Only the quantized model, not the reference one

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = validation.input_ids[:, begin_loc:end_loc].to(DEVICE)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        ts = time.perf_counter()
        quantized_outputs = quantized_model(input_ids, labels=target_ids, return_dict=True)
        te = time.perf_counter()
        inference_time += te - ts
        
        unquantized_outputs = unquantized_model(input_ids, labels=target_ids, return_dict=True)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = quantized_outputs.loss

    # Accumulate the total negative log-likelihood and the total number of tokens
    num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
    batch_size = target_ids.size(0)
    num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
    nll_sum += neg_log_likelihood * num_loss_tokens
    kld_sum += torch.nn.KLDivLoss(reduction="batchmean")(
        torch.nn.functional.log_softmax(quantized_outputs.logits.float()[0][-num_valid_tokens:, :], dim=1),
        torch.nn.functional.softmax(unquantized_outputs.logits.float()[0][-num_valid_tokens:, :], dim=1)
    )
    n_tokens += num_loss_tokens
    n_runs += 1

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
avg_kld = kld_sum / n_runs
ppl = torch.exp(avg_nll)
print(f"Perplexity: {ppl}")
print(f"KLD: {avg_kld}")
print(f"Inference time: {inference_time}")
