from opencompass.models.base import BaseModel
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from typing import Dict, List, Optional, Union
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger


os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

########################################################################################################

# MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

# print(f'Loading model - {MODEL_NAME}')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16', verbose=False)
# pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)



class RWKV6(BaseModel):
    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 16,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
    ):  # noqa
        # if tokenizer_only:
        #     self.pipeline = PIPELINE(path, tokenizer_path)
        # else:
        self._load_model(path=path,tokenizer_path=tokenizer_path)
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self,
                    path: str,
                    tokenizer_path: Optional[str] = None):
        
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE
        self.model = RWKV(model=path, strategy='cuda fp16', verbose=False)
        self.pipeline = PIPELINE(path, tokenizer_path)
        self.tokenizer = TokenizerWrapper(self.pipeline.tokenizer)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        tokens = self.tokenizer.encode(prompt)
        return len(tokens)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        all_tokens = []
        out_last = 0
        out_str = ''
        state = None
        for i in range(max_out_len):

            # forward & adjust prob.
            tokens = self.tokenizer.encode(inputs[0]) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:512], state)
                tokens = tokens[512:]
                
            # for n in args.token_ban:
            #     out[n] = -float('inf')
     
            # sampler
            token = self.pipeline.sample_logits(out, temperature=1, top_p=0.3)
            # if token in args.token_stop:
            #     break
            all_tokens += [token]

            
            ttt = self.tokenizer.decode([token])
            www = 1
            if ttt in ' \t0123456789':
                www = 0
      
            # output
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                # if callback:
                #     callback(tmp)
                out_str += tmp
                out_last = i + 1
        return [out_str]

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)
        # params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # tokenize
        prompt_tokens = [self.tokenizer.encode(x) for x in inputs]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = max_prompt_size
        tokens = torch.zeros((bsz, total_len)).long()
        for k, t in enumerate(prompt_tokens):
            num_token = min(total_len, len(t))
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        # forward
        outputs,_ = self.model.forward(tokens[0],None,full_output=True)
        # compute ppl
        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = tokens[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits, shift_labels.cuda()).view(bsz, -1)
        lens = (tokens != 0).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        torch.cuda.empty_cache()
        return ce_loss