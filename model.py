# model.py

import os.path
import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, DynamicCache
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

def merge_data(data):
    merged_data = []

    # 用于记录每个子列表开始的位置
    start_positions = []

    # 当前起始位置
    current_position = 0

    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions

def stack_and_pad_right(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat([torch.ones(tensor.shape[0], dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks

def stack_and_pad_left(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                 torch.ones(tensor.shape[0], dtype=torch.long)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=False,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
)

class LogLLM(nn.Module):
    def __init__(self, Bert_path, Llama_path, ft_path=None, is_train_mode=True, device = torch.device("cuda:0"), max_content_len = 128, max_seq_len = 128):
        super().__init__()
        self.max_content_len = max_content_len  # max length of each log messages (contents)
        self.max_seq_len = max_seq_len   # max length of each log sequence  (log sequence contains some log messages)
        self.device = device


        # self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")

        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="left", use_fast=True, trust_remote_code=True)
        
        
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token

        self.Llama_model = AutoModelForCausalLM.from_pretrained(Llama_path, quantization_config=bnb_config,
                                                           low_cpu_mem_usage=True,
                                                           device_map=device)  # embedding dim = 4096

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)
        self.Bert_model = BertModel.from_pretrained(Bert_path, quantization_config=bnb_config, low_cpu_mem_usage=True,
                                               device_map=device)

        
        # self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size, device=device)
        # self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size).half().to(device)

        # 替换 projector 定义 for Qwen3-Coder-30B-A3B-Instruct
        self.projector = nn.Sequential(
            nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size, device=device),
            nn.GELU(),
            nn.LayerNorm(self.Llama_model.config.hidden_size, device=device),
            nn.Linear(self.Llama_model.config.hidden_size, self.Llama_model.config.hidden_size, device=device)
        )


        # pre_prompt = 'Below is a sequence of system log messages: '
        # pre_prompt = "Below is a sequence of logs from the Virtual Control Center service of a power grid's Master Terminal Unit, communicating via the IEC 60870-5-104 protocol:"
        
        pre_prompt = "Below is a sequence of wind turbine SCADA sensor data and status information: "
        
        post_prompt = '. Is this sequence normal or anomalous? \n'

#         pre_prompt = """Below is a sequence of ICS communication log entries from the IEC 60870-5-104 protocol:
# {
#     "fields": [
#         "TimeStamp",
#         "Relative Time",
#         "srcIP",
#         "dstIP",
#         "srcPort",
#         "dstPort",
#         "ipLen",
#         "len",
#         "fmt",
#         "uType",
#         "asduType",
#         "numix",
#         "cot",
#         "oa",
#         "addr",
#         "ioa"
#     ],
#     "data": [
# """

#         pre_prompt = """Below is a sequence of wind turbine SCADA sensor data and status logs recorded at 10-minute intervals:
# {
#     "fields": [
#         "Timestamp",
#         "Generator RPM Max. [RPM]",
#         "Generator RPM Min. [RPM]",
#         "Generator RPM Avg. [RPM]",
#         "Generator RPM StdDev [RPM]",
#         "Generator Bearing Temp. Avg. [°C]",
#         "Generator Phase1 Temp. Avg. [°C]",
#         "Generator Phase2 Temp. Avg. [°C]",
#         "Generator Phase3 Temp. Avg. [°C]",
#         "Generator SlipRing Temp. Avg. [°C]",
#         "Generator Bearing2 Temp. Avg. [°C]",
#         "Generator CoolingWater Temp. Avg. [°C]",
#         "Hydraulic Oil Temp. Avg. [°C]",
#         "Gear Oil TemperatureBasis Avg. [°C]",
#         "Gear Oil TemperatureLevel1 Avg. [°C]",
#         "Gear Oil TemperatureLevel2_3 Avg. [°C]",
#         "Gear Bearing TemperatureHSRotorEnd Avg. [°C]",
#         "Gear Bearing TemperatureHSGeneratorEnd Avg. [°C]",
#         "Gear Bearing TemperatureHSMiddle Avg. [°C]",
#         "Gear Bearing TemperatureHollowShaftRotor Avg. [°C]",
#         "Gear Bearing TemperatureHollowShaftGenerator Avg. [°C]",
#         "Nacelle Temp. Avg. [°C]",
#         "Avg. direction [°]",
#         "Rotor RPM Max. [RPM]",
#         "Rotor RPM Min. [RPM]",
#         "Rotor RPM Avg. [RPM]",
#         "Rotor RPM StdDev [RPM]",
#         "Ambient WindSpeed Max. [m/s]",
#         "Ambient WindSpeed Min. [m/s]",
#         "Ambient WindSpeed Avg. [m/s]",
#         "Ambient WindSpeed StdDev [m/s]",
#         "Ambient WindDir Relative Avg. [°]",
#         "Ambient WindDir Absolute Avg. [°]",
#         "Ambient Temp. Avg. [°C]",
#         "Ambient WindSpeed Estimated Avg. [m/s]",
#         "Grid InverterPhase1 Temp. Avg. [°C]",
#         "Grid RotorInvPhase1 Temp. Avg. [°C]",
#         "Grid RotorInvPhase2 Temp. Avg. [°C]",
#         "Grid RotorInvPhase3 Temp. Avg. [°C]",
#         "Grid Production Power Avg. [W]",
#         "Grid Production CosPhi Avg.",
#         "Grid Production Frequency Avg. [Hz]",
#         "Grid Production VoltagePhase1 Avg. [V]",
#         "Grid Production VoltagePhase2 Avg. [V]",
#         "Grid Production VoltagePhase3 Avg. [V]",
#         "Grid Production CurrentPhase1 Avg. [A]",
#         "Grid Production CurrentPhase2 Avg. [A]",
#         "Grid Production CurrentPhase3 Avg. [A]",
#         "Grid Production Power Max. [W]",
#         "Grid Production Power Min. [W]",
#         "Grid Busbar Temp. Avg. [°C]",
#         "Grid Production Power StdDev [W]",
#         "Grid Production ReactivePower Avg. [W]",
#         "Grid Production ReactivePower Max. [W]",
#         "Grid Production ReactivePower Min. [W]",
#         "Grid Production ReactivePower StdDev [W]",
#         "Grid Production PossiblePower Avg. [W]",
#         "Grid Production PossiblePower Max. [W]",
#         "Grid Production PossiblePower Min. [W]",
#         "Grid Production PossiblePower StdDev [W]",
#         "Grid Production PossibleInductive Avg. [var]",
#         "Grid Production PossibleInductive Max. [var]",
#         "Grid Production PossibleInductive Min. [var]",
#         "Grid Production PossibleInductive StdDev [var]",
#         "Grid Production PossibleCapacitive Avg. [var]",
#         "Grid Production PossibleCapacitive Max. [var]",
#         "Grid Production PossibleCapacitive Min. [var]",
#         "Grid Production PossibleCapacitive StdDev [var]",
#         "Active power limit [W]",
#         "Active power limit source",
#         "Reactive power set point [var]",
#         "Power factor set point",
#         "Power factor set point source",
#         "Controller Ground Temp. Avg. [°C]",
#         "Controller Top Temp. Avg. [°C]",
#         "Controller Hub Temp. Avg. [°C]",
#         "Controller VCP Temp. Avg. [°C]",
#         "Controller VCP ChokecoilTemp. Avg. [°C]",
#         "Controller VCP WaterTemp. Avg. [°C]",
#         "Spinner Temp. Avg. [°C]",
#         "Spinner Temp. SlipRing Avg. [°C]",
#         "Blades PitchAngle Min. [°]",
#         "Blades PitchAngle Max. [°]",
#         "Blades PitchAngle Avg. [°]",
#         "Blades PitchAngle StdDev [°]",
#         "HVTrafo Phase1 Temp. Avg. [°C]",
#         "HVTrafo Phase2 Temp. Avg. [°C]",
#         "HVTrafo Phase3 Temp. Avg. [°C]",
#         "HVTrafo AirOutlet Temp. Avg. [°C]",
#         "HourCounters Average Total Avg. [h]",
#         "HourCounters Average GridOn Avg. [h]",
#         "HourCounters Average GridOk Avg. [h]",
#         "HourCounters Average TurbineOk Avg. [h]",
#         "HourCounters Average Run Avg. [h]",
#         "HourCounters Average Gen1 Avg. [h]",
#         "HourCounters Average Gen2 Avg. [h]",
#         "HourCounters Average Yaw Avg. [h]",
#         "HourCounters Average ServiceOn Avg. [h]",
#         "HourCounters Average AmbientOk Avg. [h]",
#         "HourCounters Average WindOk Avg. [h]",
#         "HourCounters Average AlarmActive Avg. [h]",
#         "Total hour counter [h]",
#         "Grid on hours [h]",
#         "Grid ok hours [h]",
#         "Turbine ok hours [h]",
#         "Run hours [h]",
#         "Generator 1 hours [h]",
#         "Generator 2 hours [h]",
#         "Yaw hours [h]",
#         "Service hours [h]",
#         "Ambient ok hours [h]",
#         "Wind ok hours [h]",
#         "Production LatestAverage Active Power Gen 0 Avg. [W]",
#         "Production LatestAverage Active Power Gen 1 Avg. [W]",
#         "Production LatestAverage Active Power Gen 2 Avg. [W]",
#         "Production LatestAverage Total Active Power Avg. [W]",
#         "Production LatestAverage Reactive Power Gen 0 Avg. [var]",
#         "Production LatestAverage Reactive Power Gen 1 Avg. [var]",
#         "Production LatestAverage Reactive Power Gen 2 Avg. [var]",
#         "Production LatestAverage Total Reactive Power Avg. [var]",
#         "Active power generator 0, Total accumulated [W]",
#         "Active power generator 1, Total accumulated [W]",
#         "Active power generator 2, Total accumulated [W]",
#         "Total Active power [W]",
#         "Reactive power generator 0,Total accumulated [var]",
#         "Reactive power generator 1, Total accumulated [var]",
#         "Reactive power generator 2, Total accumulated [var]",
#         "Total reactive power [var]",
#     ],
#     "data": [
# """

#         post_prompt = """    ]
# }
# Is this sequence normal or anomalous? \n
# """

        self.instruc_tokens = self.Llama_tokenizer(
            [pre_prompt, post_prompt],
            return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

        # if is_train_mode:
        #     self.Bert_model = prepare_model_for_kbit_training(self.Bert_model)
        #     self.Llama_model = prepare_model_for_kbit_training(self.Llama_model)

        if ft_path is not None:
            print(f'Loading peft model from {ft_path}.')
            Llama_ft_path = os.path.join(ft_path, 'Llama_ft')
            Bert_ft_path = os.path.join(ft_path, 'Bert_ft')
            projector_path = os.path.join(ft_path, 'projector.pt')
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.Bert_model = PeftModel.from_pretrained(
                self.Bert_model,
                Bert_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
        else:
            print(f'Creating peft model.')
            Bert_peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                          r=4,
                                          lora_alpha=32,
                                          lora_dropout=0.01)
            self.Bert_model = get_peft_model(self.Bert_model, Bert_peft_config)


            # Llama_peft_config = LoraConfig(
            #     r=8,
            #     lora_alpha=16,
            #     lora_dropout=0.1,
            #     target_modules=["q_proj", "v_proj"],
            #     bias="none",
            #     task_type=TaskType.CAUSAL_LM
            # )

            # 调参数 for Qwen3-Coder-30B-A3B-Instruct
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
        Llama_ft_path = os.path.join(path,'Llama_ft')
        Bert_ft_path = os.path.join(path,'Bert_ft')
        projector_path = os.path.join(path,'projector.pt')
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization = True)
        self.Bert_model.save_pretrained(Bert_ft_path, safe_serialization =True)
        torch.save(self.projector.state_dict(), projector_path)


    def set_train_only_projector(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_train_only_Llama(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = False
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def set_train_projectorAndBert(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False


    def set_finetuning_all(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True


    def train_helper(self, inputs, seq_positions, labels):
        '''
        :param inputs: the tokenized Sequences for BERT. Sequences are concatenated.
        :param: seq_positions:
        :param labels: np.array of labels, label is one of ['anomalous', 'normal']
        :return: Llama_output[label_mask], target_tokens_ids[target_tokens_atts]
        '''
        batch_size = len(labels)


        outputs = self.Bert_model(**inputs).pooler_output  # dim = 768
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()

        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels = np.char.add(np.char.add(prefix, labels.astype(f'U{max_len}')), ".")


        # answer_tokens = self.Llama_tokenizer(list(labels), padding=True, return_tensors="pt").to(self.device)
        # target_tokens_ids = torch.cat([answer_tokens['input_ids'][:, 1:],
        #                                torch.full((batch_size, 1), self.Llama_tokenizer.eos_token_id, device=self.device)],
        #                               dim=-1)  # add eos token
        # target_tokens_atts = answer_tokens['attention_mask'].bool()

        # answer_tokens_ids = answer_tokens['input_ids'][:, 1:]  # remove bos token
        # answer_tokens_atts = answer_tokens['attention_mask'].bool()[:, 1:]

        # for Qwen3-Coder-30B-A3B-Instruct
        answer_tokens = self.Llama_tokenizer(
            list(labels),
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


        if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = self.Llama_model.model.model.embed_tokens(answer_tokens_ids)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = self.Llama_model.model.embed_tokens(answer_tokens_ids)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        embeddings = []
        target_lens = []
        for seq_embedding, answer_embedding, answer_tokens_att in zip(seq_embeddings, answer_embeddings,
                                                                      answer_tokens_atts):
            full_prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_embedding[answer_tokens_att]])
            target_lens.append(answer_tokens_att.sum())
            embeddings.append(full_prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        attention_mask = attention_mask.to(self.device)
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, :-target_lens[i]-1] = 0
        label_mask = label_mask.bool()

        Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        return Llama_output[label_mask], target_tokens_ids[target_tokens_atts]

    def forward(self, inputs, seq_positions):
        '''
        :param inputs: the tokenized Sequences for BERT. Sequences are concatenated.
        :param seq_positions:
        :return: Generated answer (token id).
        '''
        batch_size = len(seq_positions) + 1

        outputs = self.Bert_model(**inputs).pooler_output  # dim = 768
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()

        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        prefix = "The sequence is"


        # answer_prefix_tokens = self.Llama_tokenizer(prefix, padding=True, return_tensors="pt")['input_ids'][0,1:].to(
        #     self.device)
        
        # for Qwen3-Coder-30B-A3B-Instruct
        answer_prefix_tokens = self.Llama_tokenizer(
            prefix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(self.device)


        if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.model.embed_tokens(answer_prefix_tokens)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.embed_tokens(answer_prefix_tokens)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]



        promot_embeddings = []
        for seq_embedding in seq_embeddings:
            prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_prefix_tokens_embeddings])
            promot_embeddings.append(prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(promot_embeddings)
        attention_mask = attention_mask.to(self.device)

        pad_token_id = self.Llama_tokenizer.pad_token_id
        eos_token_id = self.Llama_tokenizer.eos_token_id



        # generated_ids = self.Llama_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=6,         # 原来你最多生成 6 个
        #     do_sample=False,          # 保持 greedy
        #     eos_token_id=eos_token_id,
        #     pad_token_id=pad_token_id,
        # )

        # # 只保留新生成的部分（最后 6 个 token）
        # return generated_ids[:, -6:]



        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.device) if eos_token_id is not None else None

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        this_peer_finished = False
        answer = []
        past_key_values = DynamicCache()  # 新缓存对象


        while not this_peer_finished:
            if len(past_key_values) == 0:
                # 初始轮：传完整 inputs_embeds
                outputs = self.Llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # 后续轮：只传一个 token 的 embedding（即上一步预测的 token）
                outputs = self.Llama_model(
                    inputs_embeds=next_tokens_embeddings[:, None, :],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # 应对结束符逻辑
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            answer.append(next_tokens)

            # obtain embedding of next token
            if isinstance(self.Llama_model, peft.peft_model.PeftModelForCausalLM):
                next_tokens_embeddings = self.Llama_model.model.model.embed_tokens(next_tokens)
            else:
                next_tokens_embeddings = self.Llama_model.model.embed_tokens(next_tokens)

            # update attention_mask
            attention_mask = torch.cat([attention_mask, unfinished_sequences[:, None]], dim=1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum answer length
            if  5 < len(answer):
                this_peer_finished = True

        return torch.stack(answer,dim=1)
