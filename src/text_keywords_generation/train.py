from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config, TrainingArguments, Trainer

import torch
import os
import argparse
import random
import numpy as np

import sys
sys.path.append('/home/user/project/text_generation/')

from src.util import read_data, split_data
from src.text_keywords_generation.dataset import GetDataset, get_train_val_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    设置seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_tokenizer(tokenizer_path, special_token_path=None):
    '''
    加载tokenizer
    :param tokenizer_path:
    :param special_token_path:
    :return:
    '''
    print('tokenizer loadding...')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if special_token_path:
        tokenizer.add_special_tokens(special_token_path)
    return tokenizer

def load_pretrained_mode(tokenizer, pretrained_model_path, special_token_path=None):
    '''
    加载 pretrained model
    :param tokenizer:
    :param pretrained_model_path:
    :param special_token_path:
    :return:
    '''
    print("pretrained model loadding...")
    gpt2Config = GPT2Config.from_pretrained(pretrained_model_path,
                                            bos_token_id=tokenizer.bos_token,
                                            eos__token_id=tokenizer.eos_token,
                                            sep_token_id=tokenizer.sep_token,
                                            pad_token_id=tokenizer.pad_token,
                                            output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_path, config=gpt2Config)

    if special_token_path:
        # 添加special token,model embedding size需要作调整
        model.resize_token_embeddings(len(tokenizer))

    '''
    # bias和layernorm.weight不衰减，其它衰减
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        
        '''

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 1.只训练最后6个block
    '''
    for i, m in enumerate(model.transformer.h):
        if (i + 1) > 6:
            for param in m.parameters():
                param.requires_grad=True
    '''
    # 2.或者只训练最后的一层
    for param in model.lm_head.parameters():
        param.requires_grad=True


    return model.to(DEVICE)

def build_mode(tokenizer, model_config, special_token_path=None):
    '''
    未使用pretrained model
    :param tokenizer:
    :param model_config:
    :param special_token_path:
    :return:
    '''
    gpt2Config = GPT2Config.from_json_file(model_config)
    model = GPT2LMHeadModel(config=gpt2Config)

    if special_token_path:
        model.resize_token_embeddings(len(tokenizer))
    return model.to(DEVICE)

def train_val(model, tokenizer, train_dataset, val_dataset, param_args):
    '''
    训练
    :param model:
    :param tokenizer:
    :param train_dataset
    :param val_dataset
    :param param_args
    :return:
    '''
    
    training_args = TrainingArguments(output_dir=param_args.output_dir,
                                      num_train_epochs=param_args.epochs,
                                      per_device_train_batch_size=param_args.batch_size,
                                      per_device_eval_batch_size=len(val_dataset),
                                      gradient_accumulation_steps=param_args.gradient_accumulation_steps,
                                      evaluation_strategy=param_args.evaluation_strategy,
                                      fp16=param_args.fp16,
                                      fp16_opt_level=param_args.apex_opt_level,
                                      warmup_steps=param_args.warmup_steps,
                                      learning_rate=param_args.lr,
                                      adam_epsilon=param_args.adam_eps,
                                      weight_decay=param_args.weight_decay,
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      logging_dir=param_args.logging_dir,
                                      )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("path : {}".format(path))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, "model/pretrained_model_401"),
        type=str,
        required=False,
        help='预训练模型路径'
    )
    parser.add_argument(
        "--config_path",
        default=os.path.join(path, "model/pretrained_model_401/config.json"),
        type=str,
        required=False,
        help="模型参数",
    )
    parser.add_argument(
        '--special_token_path',
        default=os.path.join(path, 'model/pretrained_model_401/special_tokens_map.json')
    )
    parser.add_argument(
        "--vocab_path",
        default=os.path.join(path, "model/pretrained_model_401/vocab.txt"),
        type=str,
        required=False,
        help="选择词典",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(path, 'data/news.csv'),
        type=str,
        required=False,
        help="训练语料",
    )
    parser.add_argument("--epochs", default=10, type=int, required=False, help="训练epochs")
    parser.add_argument(
        "--batch_size", default=8, type=int, required=False, help="训练batch size"
    )
    parser.add_argument("--lr", default=1.5e-3, type=float, required=False, help="学习率")
    parser.add_argument("--warmup_steps", default=1e2, type=float, required=False, help="lr更新的耐心系数")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False, help="多少次更新一次梯度")
    parser.add_argument("--weight_decay", default=1e-2, type=float, required=False, help="衰减系数")
    parser.add_argument(
        "--max_length", default=768, type=int, required=False, help="单条文本最长长度"
    )
    parser.add_argument(
        "--train_ratio", default=0.9, type=float, required=False, help="训练集比例"
    )
    parser.add_argument(
        "--print_loss", default=1, type=int, required=False, help="多少步打印一次loss"
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(path, 'model/text_keywords_generation_model'), type=str, required=False, help="模型输出路径"
    )
    parser.add_argument("--logging_dir", default=os.path.join(path, 'model/text_keywords_generation_model/logs'), type=str, required=False, help="log输入路径")
    parser.add_argument(
        "--seed", default=2021, type=int, required=False, help="python hash seed"
    )
    parser.add_argument(
        "--use_apex", default=True, type=bool, required=False, help="使用apex"
    )
    parser.add_argument("--fp16", default=True, type=bool, required=False, help="使用apex单精度")
    parser.add_argument("--evaluation_strategy", default="epoch", type=str, required=False, help="评估策略")
    parser.add_argument("--adam_eps", default=1e-8, type=float, required=False, help="adam eps，防止除零")
    parser.add_argument("--apex_opt_level", default="o1", type=str, required=False, help="apex训练类型")

    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model_path
    config_path = args.config_path
    vocab_path = args.vocab_path
    data_path = args.data_path
    special_token_path = args.special_token_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    max_length = args.max_length
    train_ratio = args.train_ratio
    print_loss = args.print_loss
    output_dir = args.output_dir
    logging_dir = args.logging_dir
    seed = args.seed
    use_apex = args.use_apex
    apex_opt_level = args.apex_opt_level
    warmup_steps = args.warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    fp16 =  args.fp16
    evaluation_strategy = args.evaluation_strategy
    
    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]",
     "bos_token": "[BOS]", "eos_token": "[EOS]"}

    # train data format
    columns = [
        'title',
        'keywords',
        'content'
    ]

    # read data
    pd_data = read_data(data_path, columns)

    # split train and val
    train_set, val_set = split_data(pd_data, 0.9)

    # load tokenize
    tokenizer = load_tokenizer(pretrained_model_path, SPECIAL_TOKENS)

    # 构建数据集
    trainset = GetDataset(train_set, tokenizer, max_length, SPECIAL_TOKENS)
    valset = GetDataset(val_set, tokenizer, max_length, SPECIAL_TOKENS)
    # _, _, train_dataset, val_dataset= get_train_val_dataloader(batch_size, trainset, train_ratio)

    # load pretrained model and fine tune
    #model = load_pretrained_mode(tokenizer, pretrained_model_path, SPECIAL_TOKENS)

    # build model,no pretrained model
    model = build_mode(tokenizer, config_path, SPECIAL_TOKENS)

    # train and val
    train_val(model, tokenizer, trainset, valset, args)
