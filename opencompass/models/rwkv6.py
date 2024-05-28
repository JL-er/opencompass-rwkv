from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading
import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.model import RWKV
import torch
import numpy as np

from .base import BaseModel, LMTemplateParser
from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER


PromptType = Union[PromptList, str]

class RWKV6(BaseModel):
    """Mixtral model wrapper https://github.com/open-compass/MixtralKit.

    Args:
        path (str): path to the model directory
        max_seq_len (int): max sequence length
        max_batch_size (int): max batch size
        tokenizer_only (bool): whether to load tokenizer only
        tokenizer_path (str): path to the tokenizer directory
        meta_template (dict): meta template for the model
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 4,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
        num_gpus: int = 1,
        generation_kwargs: Optional[Dict] = None
    ):  # noqa
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             tokenizer_path=tokenizer_path)
        self.max_seq_len = max_seq_len
        self.template_parser = LMTemplateParser(meta_template)
        self.logger = get_logger()
        self.args = PIPELINE_ARGS(temperature = generation_kwargs["temperature"], top_p = generation_kwargs["top_p"], top_k = generation_kwargs["top_k"], # top_k = 0 then ignore
                    alpha_frequency = generation_kwargs["alpha_frequency"],
                    alpha_presence = generation_kwargs["alpha_presence"],
                    alpha_decay = 0.996, # gradually decay the penalty
                    token_ban = [0], # ban the generation of some tokens
                    token_stop = [], # stop generation whenever you see any token here
                    chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None):
        self.model = RWKV(model=path, strategy="cuda fp16")
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.tokenizer = self.pipeline

    def _load_tokenizer(self, tokenizer_path: str):
        self.tokenizer = TRIE_TOKENIZER(tokenizer_path)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        results = []
        for input in inputs:
            generation_tokens = self.pipeline.generate(input, token_count=max_out_len, args=self.args)
        results.append(generation_tokens)
        return results
    
    def my_print(s):
        print(s, end='', flush=True)

    # def _generate(self,
    #              inputs,
    #              max_out_len: int = 512,
    #              temperature: float = 0.6) -> str:
    #     """Generate response from input prompt.

    #     Args:
    #         inputs (list): input prompt
    #         max_out_len (int): max output length
    #         temperature (float): temperature for sampling
    #     """
    #     dialogs = []
    #     for input in inputs:
    #         assert isinstance(input, (str, PromptList))
    #         if isinstance(input, str):
    #             dialog = [{'role': 'user', 'content': input}]
    #         else:
    #             dialog = []
    #             for item in input:
    #                 msg = {'content': item['prompt']}
    #                 if item['role'].upper() == 'HUMAN':
    #                     msg['role'] = 'user'
    #                 elif item['role'].upper() == 'BOT':
    #                     msg['role'] = 'assistant'
    #                 elif item['role'].upper() == 'SYSTEM':
    #                     msg['role'] = 'system'
    #                 else:
    #                     raise ValueError(f'Unknown role: {item["role"]}')
    #                 dialog.append(msg)
    #         dialogs.append(dialog)

    #     try:
    #         results = self.generator.chat_completion(
    #             dialogs,  # type: ignore
    #             max_gen_len=max_out_len,
    #             temperature=temperature,
    #         )
    #         return [r['generation']['content'] for r in results]
    #     except AssertionError:
    #         self.logger.warning('Batched data max token limit exceeded, '
    #                             'try to run one by one...')

    #     results = []
    #     for dialog in dialogs:
    #         try:
    #             result = self.generator.chat_completion(
    #                 [dialog],  # type: ignore
    #                 max_gen_len=max_out_len,
    #                 temperature=temperature,
    #             )[0]
    #             results.append(result['generation']['content'])
    #         except AssertionError:
    #             results.append('')
    #     return results

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # Tokenize the inputs
        prompt_tokens = [self.pipeline.encode(x) for x in inputs]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_prompt_size)

        ce_loss_list = []

        for i in range(bsz):
            tokens = torch.zeros((1, total_len)).cuda().long()  # Batch size of 1 for each input
            num_token = min(total_len, len(prompt_tokens[i]))
            tokens[0, :num_token] = torch.tensor(prompt_tokens[i][-num_token:]).long()

            # Generate forward & adjust prob.
            all_tokens = tokens[0].tolist()
            out, _ = self.model.forward(all_tokens, None, full_output=True)

            out = out.unsqueeze(0)  # Add batch dimension

            # Compute ppl
            shift_logits = out[..., :-1, :].contiguous().float()
            shift_labels = tokens[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # 将 logits 张量从三维展平为二维，以便与目标标签进行对齐并计算交叉熵损失。
            shift_labels = shift_labels.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
            loss = loss_fct(shift_logits, shift_labels).view(1, -1)
            lens = (tokens != 0).sum(-1).cpu().numpy()
            ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
            ce_loss_list.append(ce_loss[0])

        return np.array(ce_loss_list)

    def get_token_len(self, prompt: str) -> int:
        return len(self.pipeline.encode(prompt))