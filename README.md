# Quantization

This is a small research project to learn about model quantization. It is divided into 2 parts:  
- `quant` folder, which contains the implementation of a quantization algorithm, very competitive with other common methods
- `auto_bitrate` folder, an experimental method to automatically find optimal quantization mixes for a given bitrate

# Learnings

- Affine quantization (having a zero point along with the scale) is basically useless as the distribution of weights is centered at zero.
- It is better to have heavily quantized scales and smaller groups. At low bits you can get away with a single mantissa bit.
- KLD seems proportional to the MSE of the quantized weights all the way down to ~3.2 bits. At 2 bits the KLD shows a bigger than expected increase, possibly because the model is so quantized that it stops functioning. This also means that the KLD quadruples with every removed bit, making it very hard to get to low bits per weight.
- Bigger models are less sensitive to quantization. This is likely due to having to store less information per weight. For example, for the Qwen2.5 family:  

| Model | KLD    |  
| ----- | ------ |  
| 0.5B  | 0.122  |
| 1.5B  | 0.0895 |
| 3B    | 0.0802 |
| 7B    | 0.0446 |

- Some tensors are more sensitive to quantization than others. A small test with Qwen2.5-7B shows:

| Tensor        | Size (GB) | KLD    | %improv. | Δsize (GB) | %/Δ     |
| ------------- | --------- | ------ | -------- | ---------- | ------- |
| All IQ4_XS    | 4.052     | 0.0544 | -        | -          | -       |
| embed Q6_K    | 4.210     | 0.0540 | 0.00758  | 0.157      | 0.04814 |
| attn_v Q6_K   | 4.067     | 0.0515 | 0.05411  | 0.014      | 3.64373 |
| attn_k Q6_K   | 4.067     | 0.0537 | 0.01375  | 0.014      | 0.92607 |
| attn_q Q6_K   | 4.156     | 0.0536 | 0.01520  | 0.103      | 0.14625 |
| attn_o Q6_K   | 4.156     | 0.0544 | 0.00082  | 0.103      | 0.00794 |
| ffn_down Q6_K | 4.602     | 0.0455 | 0.16402  | 0.549      | 0.29847 |
| ffn_gate Q6_K | 4.602     | 0.0494 | 0.09273  | 0.549      | 0.16875 |
| ffn_up Q6_K   | 4.602     | 0.0480 | 0.11826  | 0.549      | 0.21520 |
| output Q6_K   | 4.210     | 0.0298 | 0.45232  | 0.157      | 2.87122 |

# Thanks

Georgi Gerganov and the llamacpp team for developing such a great engine  
Iwan Kawrakow for his work on quantization