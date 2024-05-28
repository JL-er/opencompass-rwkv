from opencompass.models import RWKV6

# Please follow the instruction in the Meta AI website https://github.com/facebookresearch/llama/tree/llama_v1
# and download the LLaMA model and tokenizer to the path './models/llama/'.
#
# The LLaMA requirement is also needed to be installed.
# *Note* that the LLaMA-2 branch is fully compatible with LLAMA-1, and the LLaMA-2 branch is used here.
#
# git clone https://github.com/facebookresearch/llama.git
# cd llama
# pip install -e .
generation_kwargs={'temperature':1.0,'top_p':0.85,'top_k':0,'alpha_frequency':0.2,'alpha_presence':0.2}
models = [
    dict(
        abbr='rwkv-1.6b',
        type=RWKV6,
        path='/home/rwkv/JL/model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
        tokenizer_path='/home/rwkv/JL/opencompass/opencompass/models/rwkv/rwkv_vocab_v20230424.txt',
        max_seq_len=2048,
        max_batch_size=1,
        batch_size=1,
        num_gpus=1,
        generation_kwargs=generation_kwargs,
    ),
]