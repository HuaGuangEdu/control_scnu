# 作者：tomoya
# 创建：2022-10-01
# 更新：2022-10-01
# 用意：自动生成古诗
import numpy as np
import re
from paddle.nn import Layer
import paddle
from control.util.all_path import model_path
import os
from paddlenlp.transformers import BertTokenizer, BertModel, BertForTokenClassification
from control.util.download import download, getFileSize, models
# 查看模型是否存在且完整
if os.path.exists(os.path.join(model_path, "autoPoetry.pdparams")) is False or getFileSize(
        os.path.join(model_path, "autoPoetry.pdparams")) != models['自动生成古诗']["actual_size"]:
    # 没有本地化语音的模型，所以要下载模型
    print("未发现模型或模型不完整，准备下载模型")
    download("自动生成古诗")


class PoetryBertModel(Layer):
    """
    基于BERT预训练模型的诗歌生成模型
    """

    def __init__(self, pretrained_bert_model: str, input_length: int):
        super(PoetryBertModel, self).__init__()
        bert_model = BertModel.from_pretrained(pretrained_bert_model)
        self.vocab_size, self.hidden_size = bert_model.embeddings.word_embeddings.parameters()[0].shape
        self.bert_for_class = BertForTokenClassification(bert_model, self.vocab_size)
        # 生成下三角矩阵，用来mask句子后边的信息
        self.sequence_length = input_length
        self.lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))

    def forward(self, token, token_type, input_mask, input_length=None):
        # 计算attention mask
        mask_left = paddle.reshape(input_mask, input_mask.shape + [1])
        mask_right = paddle.reshape(input_mask, [input_mask.shape[0], 1, input_mask.shape[1]])
        # 输入句子中有效的位置
        mask_left = paddle.cast(mask_left, 'float32')
        mask_right = paddle.cast(mask_right, 'float32')
        attention_mask = paddle.matmul(mask_left, mask_right)
        # 注意力机制计算中有效的位置
        if input_length is not None:
            lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))
        else:
            lower_triangle_mask = self.lower_triangle_mask
        attention_mask = attention_mask * lower_triangle_mask
        # 无效的位置设为极小值
        attention_mask = (1 - paddle.unsqueeze(attention_mask, axis=[1])) * -1e10
        attention_mask = paddle.cast(attention_mask, self.bert_for_class.parameters()[0].dtype)

        output_logits = self.bert_for_class(token, token_type_ids=token_type, attention_mask=attention_mask)

        return output_logits

class PoetryGen(object):
    """
    定义一个自动生成诗句的类，按照要求生成诗句
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成诗句的最大长度，需小于等于model所允许的最大长度
    """

    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.puncs = ['，', '。', '？', '；']
        self.max_length = max_length

    def generate(self, style='', head='', topk=2):
        """
        根据要求生成诗句
        style (str): 生成诗句的风格，写成诗句的形式，如“大漠孤烟直，长河落日圆。”
        head (str, list): 生成诗句的开头内容。若head为str格式，则head为诗句开始内容；
            若head为list格式，则head中每个元素为对应位置上诗句的开始内容（即藏头诗中的头）。
        topk (int): 从预测的topk中选取结果
        """
        head_index = 0
        style_ids = self.tokenizer.encode(style)['input_ids']
        # 去掉结束标记
        style_ids = style_ids[:-1]
        head_is_list = True if isinstance(head, list) else False
        if head_is_list:
            poetry_ids = self.tokenizer.encode(head[head_index])['input_ids']
        else:
            poetry_ids = self.tokenizer.encode(head)['input_ids']
        # 去掉开始和结束标记
        poetry_ids = poetry_ids[1:-1]
        break_flag = False
        while len(style_ids) + len(poetry_ids) <= self.max_length:
            next_word = self._gen_next_word(style_ids + poetry_ids, topk)
            # 对于一些符号，如[UNK], [PAD], [CLS]等，其产生后对诗句无意义，直接跳过
            if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                continue
            if head_is_list:
                if next_word in self.tokenizer.convert_tokens_to_ids(self.puncs):
                    head_index += 1
                    if head_index < len(head):
                        new_ids = self.tokenizer.encode(head[head_index])['input_ids']
                        new_ids = [next_word] + new_ids[1:-1]
                    else:
                        new_ids = [next_word]
                        break_flag = True
                else:
                    new_ids = [next_word]
            else:
                new_ids = [next_word]
            if next_word == self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                break
            poetry_ids += new_ids
            if break_flag:
                break
        return ''.join(self.tokenizer.convert_ids_to_tokens(poetry_ids))

    def _gen_next_word(self, known_ids, topk):
        type_token = [0] * len(known_ids)
        mask = [1] * len(known_ids)
        sequence_length = len(known_ids)
        known_ids = paddle.to_tensor([known_ids], dtype='int64')
        type_token = paddle.to_tensor([type_token], dtype='int64')
        mask = paddle.to_tensor([mask], dtype='float32')
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)
        # logits中对应最后一个词的输出即为下一个词的概率
        words_prob = logits[0, -1, :].numpy()
        # 依概率倒序排列后，选取前topk个词
        words_to_be_choosen = words_prob.argsort()[::-1][:topk]
        probs_to_be_choosen = words_prob[words_to_be_choosen]
        # 归一化
        probs_to_be_choosen = probs_to_be_choosen / sum(probs_to_be_choosen)
        word_choosen = np.random.choice(words_to_be_choosen, p=probs_to_be_choosen)
        return word_choosen


def poetry_show(poetry):
    pattern = r"([，。；？])"
    text = re.sub(pattern, r'\1 ', poetry)
    for p in text.split():
        if p:
            print(p)


def isAllChinese(word):
    """
    判断输入的字符串里面是否全部都是中文
    :param word:
    :return:
    """
    for ch in word:
        if not ('\u4e00' <= ch <= '\u9fff'):
            return False
    return True


# 载入已经训练好的模型
net = PoetryBertModel('bert-base-chinese', 128)
os.system('cls')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
os.system('cls')
model = paddle.Model(net)
model.load(os.path.join(model_path, "autoPoetry.pdparams"))
poetry_gen = PoetryGen(model, bert_tokenizer)
