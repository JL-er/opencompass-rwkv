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
models = [
    dict(
        abbr='rwkv-7b',
        type=RWKV6,
        path='/home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth',
        tokenizer_path='/home/rwkv/JL/opencompass/opencompass/models/rwkv/rwkv_vocab_v20230424.txt',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]