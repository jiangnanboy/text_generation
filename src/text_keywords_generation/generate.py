import os
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, TextGenerationPipeline
import torch
from src.text_keywords_generation.dataset import GetDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(tokenizer_path, SPECIAL_TOKENS=None):
    '''
    加载tokenizer
    :param tokenizer_path:
    :param SPECIAL_TOKENS:
    :return:
    '''
    print('tokenizer loadding...')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if SPECIAL_TOKENS:
        print('special tokens loadding...')
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def load_pretrained_mode(tokenizer, load_model_path, SPECIAL_TOKENS=None):
    '''
    加载 pretrained model
    :param tokenizer:
    :param load_model_path:
    :param SPECIAL_TOKENS:
    :return:
    '''
    print("pretrained model loadding...")
    if SPECIAL_TOKENS:
        gpt2Config = GPT2Config.from_pretrained(load_model_path,
                                                bos_token_id=tokenizer.bos_token_id,
                                                eos__token_id=tokenizer.eos_token_id,
                                                sep_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                output_hidden_states=False)
    else:
        gpt2Config = GPT2Config.from_pretrained(load_model_path,
                                                pad_token_id=tokenizer.pad_token_id,
                                                output_hidden_states=False)

    model = GPT2LMHeadModel.from_pretrained(load_model_path, config=gpt2Config)

    if SPECIAL_TOKENS:
        # 添加special token,model embedding size需要作调整
        model.resize_token_embeddings(len(tokenizer))

    '''
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
    '''

    return model

def finetune_model_generate(model_path, SPECIAL_TOKENS, MAX_LEN):
    '''
    Finetune model is used to generate text!
    :param model_path:
    :param SPECIAL_TOKENS
    :param MAX_LEN
    :return:
    '''
    tokenizer = load_tokenizer(model_path)
    model = load_pretrained_mode(tokenizer, model_path, SPECIAL_TOKENS)
    while True:
        input_title = input("input title:")
        input_keywords = input("input keywords:")
        keywords = GetDataset.join_keywords(input_keywords)
        input_text = SPECIAL_TOKENS['bos_token'] + input_title + SPECIAL_TOKENS['sep_token'] + keywords + \
                      SPECIAL_TOKENS['sep_token']
        input_text_encoder = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
        model.to(DEVICE)
        input_text_encoder = input_text_encoder.to(DEVICE)
        model.eval()
        output_text = model.generate(input_text_encoder,
                                     do_sample=True,
                                     min_length=50,
                                     max_length=MAX_LEN,
                                     top_k=30,
                                     top_p=0.7,
                                     temperature=0.9,
                                     repetition_penalty=2.0,
                                     num_return_sequences=10)
        '''
        # beam-search
        output_text = model.generate(input_text_encoder,
                                     do_sample=True,
                                     max_length=MAX_LEN,
                                     num_beams=5,
                                     repetition_penalty=5.0,
                                     early_stopping=True,
                                     num_return_sequences=1)
        '''
        for i, output in enumerate(output_text):
            text = tokenizer.decode(output, skip_special_tokens=True)
            text_len = len(input_title) + len(','.join(input_keywords))
            print('{}: {}\n'.format(i + 1, text[text_len:]))

def raw_model_generate(model_path):
    '''
    Raw model(no finetune) is used to generate text!
    :param model_path:
    :return:
    '''
    print("model and tokenizer loading...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    text_generator = TextGenerationPipeline(model, tokenizer)
    while True:
        input_text = input("input text:")
        if (input_text == 'exit') or (input_text == ""):
            break
        else:
            output_text = text_generator(input_text, max_length=100, do_sample=True)
            print(output_text[0]["generated_text"])


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    model_path = os.path.join(path, "model/gpt2_5g")

    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}

    finetune_model_generate(model_path, SPECIAL_TOKENS, 1024)

