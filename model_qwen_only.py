# model_qwen_only.py

import os.path
import peft
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, DynamicCache
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

def stack_and_pad_left(tensors, pad_value=0):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用 pad_value 填充 (默认是0，但对于input_ids可能需要是pad_token_id)
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0), value=pad_value)
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码 (1为真实数据，0为padding)
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=tensor.device),
                                 torch.ones(tensor.shape[0], dtype=torch.long, device=tensor.device)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class LogLLM(nn.Module):
    def __init__(self, Llama_path, ft_path=None, is_train_mode=True, device=torch.device("cuda:0"), max_content_len=128, max_seq_len=128):
        super().__init__()
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.device = device

        print(f"Loading Qwen/Llama model from: {Llama_path}")
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="left", use_fast=True, trust_remote_code=True)
        
        if self.Llama_tokenizer.pad_token is None:
            self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
            self.Llama_tokenizer.pad_token_id = self.Llama_tokenizer.eos_token_id

        self.Llama_model = AutoModelForCausalLM.from_pretrained(Llama_path, quantization_config=bnb_config,
                                                               low_cpu_mem_usage=True,
                                                               device_map=device)

        # 定义 Prompt
        pre_prompt = 'Below is a sequence of IEC-104 protocol communication logs:'
        post_prompt = '. Is this sequence normal or anomalous? \n'

        self.instruc_tokens = self.Llama_tokenizer(
            [pre_prompt, post_prompt],
            return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

        if ft_path is not None:
            print(f'Loading peft model from {ft_path}.')
            # 仅加载 Llama 的 LoRA
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                ft_path, # 直接指向 ft_path，因为不需要区分 Bert_ft 和 Llama_ft
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
        else:
            print(f'Creating peft model for Qwen/Llama.')
            # 针对 Qwen/Llama 的 LoRA 配置
            Llama_peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"
                ],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)

    def save_ft_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # 直接保存，不需要分别保存 Bert/Projector
        self.Llama_model.save_pretrained(path, safe_serialization=True)

    def set_finetuning_all(self):
        # 只需要开启 LoRA 参数
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train_helper(self, inputs, seq_positions, labels):
        '''
        :param inputs: tokenized Log Contents by *Llama_tokenizer*. 
                       Currently flat (batch of log lines).
        :param seq_positions: split points for sequences.
        :param labels: ['anomalous', 'normal', ...]
        '''
        batch_size = len(labels)

        # 1. 获取日志内容的 Embedding (直接用 Qwen embedding)
        # inputs 应该是 {'input_ids': ..., 'attention_mask': ...}
        input_ids = inputs['input_ids']
        
        # 获取 Embedding
        if isinstance(self.Llama_model, peft.peft_model.PeftModelForCausalLM):
            base_model = self.Llama_model.model.model
        else:
            base_model = self.Llama_model.model

        # [Total_Log_Lines, Hidden_Size]
        log_embeddings = base_model.embed_tokens(input_ids)

        # 2. 根据 seq_positions 将拍平的日志 Embedding 切分为 Sequences
        # seq_embeddings 是一个 tuple，每个元素是一个 sample 的所有日志 embeddings [Num_Logs_In_Seq, Len, Hidden]
        seq_embeddings_list = torch.tensor_split(log_embeddings, seq_positions)

        # 3. 准备 Label Tokens
        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels_str = np.char.add(np.char.add(prefix, labels.astype(f'U{max_len}')), ".")

        answer_tokens = self.Llama_tokenizer(
            list(labels_str),
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        target_tokens_ids = torch.cat(
            [answer_tokens["input_ids"],
             torch.full((answer_tokens["input_ids"].size(0), 1),
                        self.Llama_tokenizer.eos_token_id, device=self.device)],
            dim=-1
        )
        target_tokens_atts = torch.cat(
            [answer_tokens["attention_mask"],
             torch.ones((answer_tokens["attention_mask"].size(0), 1),
                        dtype=answer_tokens["attention_mask"].dtype, device=self.device)],
            dim=-1
        ).bool()

        answer_tokens_ids = answer_tokens["input_ids"]
        answer_tokens_atts = answer_tokens["attention_mask"].bool()
        
        answer_embeddings = base_model.embed_tokens(answer_tokens_ids)

        # 4. 准备 Instruction Embeddings
        instruc_embeddings = base_model.embed_tokens(self.instruc_tokens['input_ids'])
        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        # 5. 拼接最终的 Embedding 序列
        embeddings = []
        target_lens = []
        
        for i, (seq_embed, answer_embedding, answer_tokens_att) in enumerate(zip(seq_embeddings_list, answer_embeddings, answer_tokens_atts)):
            # seq_embed: [Num_Logs, Log_Len, Hidden] -> 需要变成 [Total_Seq_Len, Hidden]
            # 注意：这里的 Log_Len 包含了 padding，我们需要利用 inputs['attention_mask'] 去除 padding 吗？
            # 这里的 inputs 是 batch 过的，带有 padding。
            # 简单做法：直接 flatten，虽然会带入 pad token 的 embedding，但对 transformer 来说通常是可以接受的，
            # 或者更严谨的做法是把 padding mask 对应的 embedding 去掉。
            # 鉴于 CustomCollator 的实现未知，假设我们直接拼接。
            
            # seq_embed is [N, L, H]. We want [N*L, H].
            seq_embed_flat = seq_embed.view(-1, seq_embed.size(-1))
            
            # 拼接: Ins1 + Logs + Ins2 + Answer
            full_prompt_embedding = torch.cat([ins1, seq_embed_flat, ins2, answer_embedding[answer_tokens_att]])
            
            target_lens.append(answer_tokens_att.sum())
            embeddings.append(full_prompt_embedding)

        # 6. Padding & Masking
        # pad_value=0 is fine for embeddings usually, assuming 0 vector won't destroy arithmetic too much, 
        # but masked out anyway.
        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        attention_mask = attention_mask.to(self.device)
        
        # 创建 label_mask (只计算 Answer 部分的 Loss)
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            # 这里的逻辑是：倒数 target_lens[i] 个是 Answer，再减1可能是 EOS
            # label_mask 前面部分置 0
            label_mask[i, :-target_lens[i]-1] = 0
        label_mask = label_mask.bool()

        # 7. Forward
        Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        return Llama_output[label_mask], target_tokens_ids[target_tokens_atts]

    def forward(self, inputs, seq_positions):
        '''
        Inference function.
        '''
        batch_size = len(seq_positions) + 1

        input_ids = inputs['input_ids']
        
        if isinstance(self.Llama_model, peft.peft_model.PeftModelForCausalLM):
            base_model = self.Llama_model.model.model
        else:
            base_model = self.Llama_model.model

        log_embeddings = base_model.embed_tokens(input_ids)
        seq_embeddings_list = torch.tensor_split(log_embeddings, seq_positions)

        prefix = "The sequence is"
        # for Qwen/Llama
        answer_prefix_tokens = self.Llama_tokenizer(
            prefix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(self.device)
        
        answer_prefix_tokens_embeddings = base_model.embed_tokens(answer_prefix_tokens)
        instruc_embeddings = base_model.embed_tokens(self.instruc_tokens['input_ids'])
        
        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        promot_embeddings = []
        for seq_embed in seq_embeddings_list:
            seq_embed_flat = seq_embed.view(-1, seq_embed.size(-1))
            prompt_embedding = torch.cat([ins1, seq_embed_flat, ins2, answer_prefix_tokens_embeddings])
            promot_embeddings.append(prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(promot_embeddings)
        attention_mask = attention_mask.to(self.device)

        pad_token_id = self.Llama_tokenizer.pad_token_id
        eos_token_id = self.Llama_tokenizer.eos_token_id

        # Generation Loop (Greedy)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.device) if eos_token_id is not None else None

        unfinished_sequences = torch.ones(inputs_embeds.shape[0], dtype=torch.long, device=self.device)
        this_peer_finished = False
        answer = []
        past_key_values = DynamicCache()

        while not this_peer_finished:
            if len(past_key_values) == 0:
                outputs = self.Llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.Llama_model(
                    inputs_embeds=next_tokens_embeddings[:, None, :],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            answer.append(next_tokens)

            next_tokens_embeddings = base_model.embed_tokens(next_tokens)
            attention_mask = torch.cat([attention_mask, unfinished_sequences[:, None]], dim=1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if 5 < len(answer):
                this_peer_finished = True

        return torch.stack(answer, dim=1)