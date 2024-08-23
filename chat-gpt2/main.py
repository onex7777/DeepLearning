# coding=utf-8
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from datasets import load_dataset

# 使用lora进行微调，速度更快，效果和直接微调原模型效果接近（通过实验验证）。
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

# trl包，是transformer reinforcement learning的首字母缩写，提供了非常方便的transformer模型的RLHF训练API
# 感谢开源！有兴趣的可以去github上的
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb", # 这个是Huggingface上要训练的gpt2-imdb的名称，在transformer中可以用from_pretrained直接下载和缓存
    learning_rate=1.41e-5, # 学习率
    # log_with="wandb", # 使用wandb监视训练过程，也可以使用tensorboard
    log_with="tensorboard", # 使用wandb监视训练过程，也可以使用tensorboard
    # accelerator_kwargs={"logging_dir": "./tb_logger"}
)

sent_kwargs = {
    "return_all_scores": True, # 文本生成的参数，这里设置为True，表示生成文本时返回得分
    "function_to_apply": "none",
    "batch_size": 16 # 批大小，不解释了。玩深度学习这个读懂。GPU显存越大，这个可以设的越大。
}

# step 1: 使用transformers库加载模型
pretrained_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
#     device_map="auto",
#     load_in_8bit=True,
)

# 设置目标模块名称
target_modules = None
target_modules = ["c_attn"]  # workaround to use 8bit training on this model

# 设置lora配置参数
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,  # handled automatically by peft
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# step 2: 设置8bit训练
pretrained_model = prepare_model_for_int8_training(pretrained_model, output_embedding_layer_name="lm_head")
for name, param in pretrained_model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False

    if getattr(pretrained_model, "is_loaded_in_8bit", False):
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and "layer_norm" in name:
            param.data = param.data.to(torch.float16)

# step 3: 设置lora模型。做instruction learning，到这里就好了。如果要做RLHF，还要做第四步。
pretrained_model = get_peft_model(pretrained_model, lora_config)

# step 4: 将lora模型加载入trl模型，加上value head。
model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

# 做必要的设置，梯度检查。
model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

# model.gradient_checkpointing_disable()
# model.pretrained_model.config.use_cache = True
# ppo_trainer = PPOTrainer(config, model, ref_model=None,
#                          tokenizer=tokenizer, dataset=dataset,
#                          data_collator=collator)
# ppo_trainer.generate(query, **generation_kwargs)

# import wandb
# # from kaggle_secrets import UserSecretsClient
# # user_secrets = UserSecretsClient()
# # wandb_api = user_secrets.get_secret("wandb_key")
# wandb.login(key=wandb_api)
# wandb.init(project="trl_imdb_positive")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)


def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    构建训练用的数据集。使用`load_dataset`函数下载和缓存数据集。如果要用自己的数据集，则需要替换该部分代码。
    当然`load_dataset`也可以加载本地数据集，详情各自行百度，或者去datasets的官网查找帮助信息。

    Args:
        dataset_name (`str`):
            数据集名称

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            返回dataloader
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # pad_token和eos_toke3n是同一个，也可以用其它的token进行替换。
    # 加载IMDB数据集，直接从huggingface的hub上下载数据，当然也3
    # 、可以下载其他数据
    # 每次做DL或ML时，大量时间用在了做
    ds = load_dataset(dataset_name, split='train')  # 加载后是DataFrame格式！？
    ds = ds.rename_columns({'text': 'review'})
    ds = ds.filter(lambda x: len(x["review"]) >5000, batched=False)  # 这里filter是指len(x["review"])>200都过滤掉

    # 对batch_size进行裁剪，缩小到2到8之间。（2和8是函数中的默认参数）
    # 即query的token长度控制在2到8之间，有点小呀
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()]  # 后面设置batched=False,每次input_size都不同
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    # 将数值型变量设置为torch的tensor格式，并且输出所有的列数据，在RL截断需要使用！一定要注意设置output_all_columns=True
    ds.set_format(type='torch', columns=["input_ids", "label"], output_all_columns=True)
    return ds

dataset = build_dataset(config)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
"""
这里加载带有value head的GPT2模型及其对应的分词器。下面加载了两次模型；第一次加载的模型用来进行强化学习，调整参数。第二次加载的模型作为参考模型，
用来计算和前面可训练模型的KL散度。这个KL散度，用来作为PPO训练的额外奖励信号，来保证我们的模型不会太偏离原始模型（即防止灾难性遗忘情况的发生）。
"""

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model=ref_model,
                         tokenizer=tokenizer, dataset=dataset,
                         data_collator=collator)


device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)


gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

output_min_length = 8
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

"""
训练循环主要包含三个步骤：

根据query，基于GPT2生成response
拼接query和response，使用BERT来得到拼接后文本的得分
基于(query, response, reward)三元组，基于PPO算法来优化模型
训练耗时

基于上述配置，在V100上大约耗时两个小时完成训练。（如果使用peft包，可能会快一些，但是效果不知道怎么样！）
"""

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    #### Get response from gpt2
    response_tensors = []
    for i, query in enumerate(query_tensors):
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
        #  print无法打印一些特殊字符
        print("=="*20)
        try:
            print("第{}条query:{} ".format(i+1, tokenizer.decode(query)))
            print("第{}条response:{} ".format(i + 1, tokenizer.decode(response.squeeze()[-gen_len:].squeeze())))
        except:
            print("第{}条query:{} ".format(i + 1, query))
            print("第{}条response:{} ".format(i + 1, response.squeeze()[-gen_len:].squeeze()))
        print("==" * 20)
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]


    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

#     break
print("===========================test=====================================")

#### get a batch from the dataset
bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = df_batch['input_ids'].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = output_length_sampler()
    output = ref_model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
                                     max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
    response_tensors_ref.append(output) #
#     output = model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
#                                  max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
    output = ppo_trainer.generate(torch.tensor(query_tensors[i]).to(device),
                                 max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
    response_tensors.append(output)

#### decode responses
game_data['response (before)'] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data['response (after)'] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
game_data['rewards (before)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
game_data['rewards (after)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# store results in a dataframe
df_results = pd.DataFrame(game_data)


print('mean:')
print((df_results[["rewards (before)", "rewards (after)"]].mean()))
print()
print('median:')
print((df_results[["rewards (before)", "rewards (after)"]].median()))

#
# # 登录huggingface Hub
# from huggingface_hub import login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# huggingface_key = user_secrets.get_secret("huggingface_key")
# login(token=huggingface_key, add_to_git_credential=True)


model.save_pretrained('gpt2-imdb-pos-v2', push_to_hub=True)
tokenizer.save_pretrained('gpt2-imdb-pos-v2', push_to_hub=True)