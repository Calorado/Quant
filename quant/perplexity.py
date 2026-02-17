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
from transformers import BitsAndBytesConfig, HqqConfig, QuantoConfig
from transformers import Qwen2ForCausalLM
from transformers import pipeline
import torch
from tqdm import tqdm

import time
import os
import sys

def calculate_calibration_matrix(calibration, model, device):
    """Runs the model on the given calibration data to obtain calibration statistics for quantization"""
    max_length = 512  # Max length of each generation test
    stride = 128  # Sliding window stride
    seq_len = calibration.input_ids.size(1)
    prev_end_loc = 0

    print("\nCalculating calibration matrix...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = calibration.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            model(input_ids, labels=target_ids, return_dict=True)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

def load_quantized_model(
        model: str, 
        quantMethod: str,
        quantParam: str,
        calibration = None,
        device: str = "cuda"
    ):
    if quantMethod == "gguf":
        return AutoModelForCausalLM.from_pretrained(
            os.path.dirname(quantParam),
            gguf_file=quantParam,
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
    if quantMethod == "hqq":
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.core.quantize import BaseQuantizeConfig
        import hqq

        model = AutoModelForCausalLM.from_pretrained(
            model, 
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        AutoHQQHFModel.quantize_model(model, quant_config=BaseQuantizeConfig(int(quantParam)))
        return model

    if quantMethod == "quanto":
        return AutoModelForCausalLM.from_pretrained(
            model, 
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            quantization_config=QuantoConfig(weights = f"int{quantParam}"),
        )
    
    if quantMethod == "bnb":
        if quantParam == "int8":
            quantConfig = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            quantConfig = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type=quantParam, 
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        return AutoModelForCausalLM.from_pretrained(
            model, 
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            quantization_config=quantConfig,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if quantMethod == "none":
        return model

    quant = quantParam.split(";")[0]
    optimize = quantParam.split(";")[1]

    if type(model) is Qwen2ForCausalLM:
        usesTiedEmbeddings = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
        if device == "cuda":
            model.graph_buffer = torch.empty(model.model.layers[0].mlp.gate_proj.weight.nelement(), device="cuda", dtype=torch.float16)
        else:
            model.graph_buffer = None

        for layer in model.model.layers:
            layer.mlp.gate_proj = CalibrationLinear(layer.mlp.gate_proj)
            layer.mlp.up_proj = CalibrationLinear(layer.mlp.up_proj)
            layer.mlp.down_proj = CalibrationLinear(layer.mlp.down_proj)
            layer.self_attn.q_proj = CalibrationLinear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = CalibrationLinear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = CalibrationLinear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = CalibrationLinear(layer.self_attn.o_proj)
        model.lm_head = CalibrationLinear(model.lm_head)

        if optimize != OPTIMIZE_FAST:
            calculate_calibration_matrix(calibration, model, device)

        model.lm_head = QuantizedLinear(model.lm_head, quant=quant, optimize=optimize, batches=32, buffer=model.graph_buffer)
        model.model.embed_tokens = QuantizedEmbedding(
            model.lm_head if usesTiedEmbeddings else model.model.embed_tokens, 
            quant=quant, 
            optimize=optimize
        )

        for layer in model.model.layers:
            layer.mlp.gate_proj = QuantizedLinear(layer.mlp.gate_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.mlp.up_proj = QuantizedLinear(layer.mlp.up_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.mlp.down_proj = QuantizedLinear(layer.mlp.down_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.self_attn.q_proj = QuantizedLinear(layer.self_attn.q_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.self_attn.k_proj = QuantizedLinear(layer.self_attn.k_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.self_attn.v_proj = QuantizedLinear(layer.self_attn.v_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)
            layer.self_attn.o_proj = QuantizedLinear(layer.self_attn.o_proj, quant=quant, optimize=optimize, buffer=model.graph_buffer)

    return model

def get_memory_footprint(model, quantMethod: str) -> int:
    if quantMethod != "research":
        return model.get_memory_footprint()
    
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

# ------------------------------- CLI PARSING -------------------------------

if "-h" in sys.argv:
    print("Usage: \n"
          "python ./perplexity.py [OPTIONS...]\n"
          "-h show this help\n"
          "-m model: model ID to use (default Qwen/Qwen2.5-1.5B-Instruct)\n"
          "-t test: can be 'perplexity' (default) to test ppl and kld or 'generation' to test text generation\n"
          "-p prompt: prompt to use for 'generation' test (default 'tell me about Newton and his discoveries')\n"
          "-v file: validation file to use (default this script). Used by 'perplexity' test\n"
          "-c file: calibration file to use (default this script)\n"
          "-q method=param: quant method and parameters to use\n"
          "    'none': do not quantize the model\n"
          "    'gguf' file: use the given gguf file\n"
          "    'bnb' type: use bitsandbytes with FP4, NF4 or int8 type\n"
          "    'hqq' bits: use hqq with 8, 4, 3 or 2 bits per weight\n"
          "    'quanto' bits: use quanto with 8, 4 or 2 bits per weight\n"
          "    'research' param: default, use the method presented in this research. param looks like '[type];[opt]' and is 'Q4L;O2' by default\n"
          "         'type' must follow 'Q[bits][subtype]', where bits can be 8, 6, 5, 4, 3 or 2 and subtype "
          "         must be 'L' (0.5625bpw), 'M' (0.3125bpw) or 'S' (0.1875bpw)\n"
          "         'opt' must be 'O1', 'O2' or 'O3'\n"
          "-d device: device to use for inference, 'cuda' or 'cpu' (default 'cuda' if available)\n"
          )
    exit()

model = "Qwen/Qwen2.5-1.5B-Instruct"
test = "perplexity"
prompt = "tell me about Newton and his discoveries"
validation = __file__
calibration = __file__
quantMethod = "quant"
quantParam = "Q4L;O2"
device = "cuda" if torch.cuda.is_available() else "cpu"

for argi in range(1, len(sys.argv), 2):
    if sys.argv[argi] == "-t":
        test = sys.argv[argi + 1]
    elif sys.argv[argi] == "-c":
        calibration = sys.argv[argi + 1]
    elif sys.argv[argi] == "-v":
        validation = sys.argv[argi + 1]
    elif sys.argv[argi] == "-m":
        model = sys.argv[argi + 1]
    elif sys.argv[argi] == "-p":
        prompt = sys.argv[argi + 1]
    elif sys.argv[argi] == "-q":
        quantMethod = sys.argv[argi + 1].split("=")[0]
        quantParam = sys.argv[argi + 1].split("=")[1]
    elif sys.argv[argi] == "-d":
        device = sys.argv[argi + 1]
    else:
        print(f"Unknown option {sys.argv[argi]}")
        exit()

# ------------------------------- PROGRAM LOGIC -------------------------------

tokenizer = AutoTokenizer.from_pretrained(model)
text = open(validation, "r", encoding="utf-8").read()
validation = tokenizer(text, return_tensors="pt")
text = open(calibration, "r", encoding="utf-8").read()
calibration = tokenizer(text, return_tensors="pt")

ts = time.perf_counter()
quantized_model = load_quantized_model(
    model = model, 
    quantMethod = quantMethod,
    quantParam = quantParam,
    calibration = calibration,
    device = device,
)
te = time.perf_counter()

unquantized_model = AutoModelForCausalLM.from_pretrained(
    model, device_map=device, dtype=torch.float16 if device == "cuda" else torch.float32
)

print(f"Quantization time: {te - ts}")
print(f"Original model size: {get_memory_footprint(unquantized_model, None)}")
print(f"Quantized model size: {get_memory_footprint(quantized_model, quantMethod)}")

if test == "generation":
    pipeline = pipeline(
        task = "text-generation", 
        model = quantized_model, 
        tokenizer = tokenizer
    )
    ts = time.perf_counter()
    generatedText = pipeline(
        [{
            "role": "user", 
            "content": prompt,
        }], 
        max_new_tokens=2048,
        do_sample=False,
    )[0]["generated_text"][1]["content"]
    te = time.perf_counter()
    tokens = len(tokenizer.encode(generatedText))
    print(f"Generation time: {te - ts}\n"
          f"Tokens: {tokens} ({tokens / (te - ts)} tok/s)\n"
          f"{generatedText}")

elif test == "perplexity":

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
        input_ids = validation.input_ids[:, begin_loc:end_loc].to(device)
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
