from opencompass.models import RWKV6
from mmengine.config import read_base
# Please follow the instruction in the Meta AI website https://github.com/facebookresearch/llama/tree/llama_v1
# and download the LLaMA model and tokenizer to the path './models/llama/'.
#
# The LLaMA requirement is also needed to be installed.
# *Note* that the LLaMA-2 branch is fully compatible with LLAMA-1, and the LLaMA-2 branch is used here.
#
# git clone https://github.com/facebookresearch/llama.git
# cd llama
# pip install -e .
with read_base():
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_2needle_en_datasets as needlebench_multi_2needle_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_3needle_en_datasets as needlebench_multi_3needle_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_4needle_en_datasets as needlebench_multi_4needle_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_5needle_en_datasets as needlebench_multi_5needle_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_2needle_zh_datasets as needlebench_multi_2needle_zh_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_3needle_zh_datasets as needlebench_multi_3needle_zh_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_4needle_zh_datasets as needlebench_multi_4needle_zh_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_reasoning_4k import needlebench_5needle_zh_datasets as needlebench_multi_5needle_zh_datasets

    from .datasets.needlebench.needlebench_4k.needlebench_single_4k import needlebench_en_datasets as needlebench_origin_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_single_4k import needlebench_zh_datasets as needlebench_origin_zh_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_retrieval_4k import needlebench_en_datasets as needlebench_parallel_en_datasets
    from .datasets.needlebench.needlebench_4k.needlebench_multi_retrieval_4k import needlebench_zh_datasets as needlebench_parallel_zh_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
# datasets = []
# datasets += gsm8k_datasets
# datasets += math_datasets

model_meta_template = dict(
    round = [
        dict(role='HUMAN',begin='User: ',end='\n\n'),
        dict(role='BOT',begin='Assistant:',end='\n\n',generate=True),
    ]
)
models = [
    dict(
        abbr='rwkv-1.6b',
        type=RWKV6,
        path='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
        tokenizer_path='/home/rwkv/JL/opencompass/opencompass/models/rwkv/rwkv_vocab_v20230424.txt',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]