### 给定title和keywords利用gpt2生成文本
    利用gpt2，输入标题和关键词，自动生成相关文本。
    
### 数据
    数据data/news.csv，格式如下
![image](https://raw.githubusercontent.com/jiangnanboy/text_generation/master/img/data.png)

### 模型 

![image](https://raw.githubusercontent.com/jiangnanboy/text_generation/master/img/1.png)

    src/text_keywords_generation
    1.训练：src/text_keywords_generation/train.py
        linux:sh run.sh
        训练输入格式：input = self.SPECIAL_TOKENS['bos_token'] + title + self.SPECIAL_TOKENS['sep_token'] + keywords + \
                self.SPECIAL_TOKENS['sep_token'] + text + self.SPECIAL_TOKENS['eos_token']
                
        (预训练model[model/pretrained_model_401]来自：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main)
   
    2.生成:src/text_keywords_generation/generate.py
        输入：input = SPECIAL_TOKENS['bos_token'] + input_title + SPECIAL_TOKENS['sep_token'] + keywords + \
                          SPECIAL_TOKENS['sep_token']

    note:在进行fine-tune时，GPU会经常出现内存益出，应该调小batch_size等参数。
### Requirements
    requirement.txt

### References
* https://jalammar.github.io/illustrated-gpt2/
* https://github.com/Morizeyao/GPT2-Chinese
* https://github.com/huggingface/transformers
* https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
* https://colab.research.google.com/drive/1vnpMoZoenRrWeaxMyfYK4DDbtlBu-M8V?usp=sharing
    
