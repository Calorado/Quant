# Auto-Bitrate

It is well known that some tensors of an LLM are more sensitive to quantization than others, and giving more bits to those can give very large quality improvements. But most methods rely on heuristics, which might lead to suboptimal results.  

This is a short Python script that uses an optimization algorithm akin to gradient descent to try and find the best quantization mix for a given bitrate. Specifically it targets GGUF models, and leverages the existing tools from the llama.cpp project, though the algorithm itself should be applicable to any other engine or model type.

To run it will require the llama-imatrix, llama-perplexity and llama-quantize programs available and the gguf python library installed.

# Benchmarks

The model tested is Qwen3-4B-Instruct-2507. The text file used for the benchmark is validation.txt. I decided to compare Unsloth models as they are often regarded as high quality, and the ones from ByteShape, as they seem to be doing something similar to my script.

To generate my own quants I used the following command:  
`python .\auto_bitrate.py -c calibration.txt -v validation.txt -i imatrix.txt -b [BITRATE] Qwen3-4B-Instruct-2507-F16.gguf Qwen3-4B-Instruct-2507-[BITRATE]bpw.gguf`

![](quants.png)

I also made a small test with the bigger Qwen3-VL-30BA3B-Instruct. The command used is the same.

| Model        | BPW  | KLD    | Perplexity |
| ------------ | ---- | ------ | ---------- |
| Unsloth      | 3.38 | 0.055  | 5.232      |
| Auto Bitrate | 3.27 | 0.0523 | 5.253      |

# Limitations

- Quantization is very slow. A single run of Qwen3-4B took about 3 hours on my RTX 3070 Ti.
- A lot of disk space will be required to cache all the quantization levels.
- The script will write an entire model to disk to test the KLD of a single mix. Over the course of the run this will generate A LOT of write cycles. Take this into consideration for your SSD health.