# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from .parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from .parser import (remove_comments_and_docstrings,
                     tree_to_token_index,
                     index_to_code_token,
                     tree_to_variable_index)
from tree_sitter import Language, Parser
from tree_sitter import Language, Parser
import multiprocessing
from tqdm import tqdm, trange

cpu_cont = 16

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('./generator/fid/src/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


def convert_examples_to_features(item):
    js, tokenizer, args = item
    # code
    parser = parsers[args.lang]
    # extract data flow
    code_tokens, dfg = extract_dataflow(js['target'], parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]  # 一个词可能分解成多个token，每个词都以[]划分 @表示的是空格
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(
            code_tokens[i]))  # {-1: (0, 0), 0: (0, 1), 1: (1, 5), 2: (5, 6)},即一个df节点的token的位置，前闭后开
    code_tokens = [y for x in code_tokens for y in x]  # 将二维数组展开成一维
    # truncating  代码pad到最长，dataflow也是
    code_tokens = code_tokens[:args.code_length - 1] + [tokenizer.sep_token]  # 只保留code_length长度
    code_length = len(code_tokens)
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)

    # code_ids += [tokenizer.pad_token_id] * (args.code_length - code_length)
    position_idx = [i + tokenizer.pad_token_id + 2 for i in range(code_length)]  # 从2开始
    position_idx += [tokenizer.pad_token_id] * (args.code_length - code_length)

    # dfg
    dfg = dfg[:args.data_flow_length]
    code_tokens += [x[0] for x in dfg]  # 把dataflow的节点变量加入到code_token中
    position_idx += [1 for x in dfg]  # 把dataflow的节点变量加入到位置编码中，值为1
    # code_ids += [tokenizer.unk_token_id for x in dfg]  # 把dfg中的变量全部设为idx为unk_token_id(3)，加入到code_ids中
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    # code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):  # 将原本tree-sitter中给变量设置的后面的数字重新编号，从0开始
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + (
            [reverse_index[i] for i in x[-1] if i in reverse_index],)  # 将上一步得到的新的idx重新设置为dfg的comesfrom的变量后面的数字
    dfg_to_dfg = [x[-1] for x in dfg]  # 变量来自哪个变量
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]  # 变量后面的数字（数字的意义是第几个词（包括，））指向的变量的token
    # length = len([tokenizer.cls_token]) # 现在不加上<s>
    # dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]  # 全部往后面移动一位，因为在前面加了cls

    js['position_idx'] = position_idx
    js['dfg_to_code'] = dfg_to_code
    js['dfg_to_dfg'] = dfg_to_dfg
    js['code_ids'] = code_ids
    return js


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 args,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 is_train=False,
                 ):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()  # 不知道为什么
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.pool = multiprocessing.Pool(cpu_cont)
        self.args = args

        # TODO train模式下的提取dataflow
        # VER.1 全部都添加dataflow
        if self.is_train:
            self.data = self.pool.map(convert_examples_to_features, tqdm(
                [(item, self.tokenizer, self.args) for item in self.data], total=len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:  # passage:把title（函数名）text（解释）放入  返回[passage1, passage2]
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        attn_mask = None
        node_index = None
        dataflow_len = None
        # TODO: 处理code的mask
        if self.is_train:
            code_len = self.args.code_length
            # calculate graph-guided masked function
            attn_mask = np.zeros((code_len,
                                  code_len + self.args.data_flow_length), dtype=np.bool_)
            # calculate begin index of node and max length of input
            node_index = sum([i > 1 for i in example['position_idx']])  # code_token长度
            dataflow_len = sum([i == 1 for i in example['position_idx']])
            # sequence can attend to sequence
            seq_ids = torch.arange(code_len)
            causal_mask = seq_ids[None, :].repeat(code_len, 1) <= seq_ids[:, None]
            attn_mask[:, :code_len] = causal_mask
            # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(example['dfg_to_code']):
                if a < node_index and b < node_index:
                    # attn_mask[idx + code_len, a:b] = True
                    attn_mask[a:b, idx + code_len] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(example['dfg_to_dfg']):
                for a in nodes:
                    if a + code_len < len(example['position_idx']):
                        # attn_mask[idx + node_index, a + node_index] = True
                        attn_mask[example['dfg_to_code'][idx][0]:example['dfg_to_code'][idx][1], a + code_len] = True

        return {
            'index': index,
            'question': question,
            'target': target if not self.is_train else None,
            'code_ids': torch.tensor(example['code_ids']) if self.is_train else None,
            'passages': passages,
            'scores': scores,
            'decoder_mask': attn_mask,
            'node_len': node_index,
            'dataflow_len': dataflow_len
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(  # batch_encode_plus是批量转换成idx，还会返回atten_mask
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,  # 一定会pad到最大长度
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])  # [None] 可能用于在其前面添加一个额外的维度，[10,100]->[1,10,100]
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)  # [4,10,200]
    passage_masks = torch.cat(passage_masks, dim=0)  # [4,10,200]
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, code_maxlength, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.code_maxlength = code_maxlength

    def __call__(self, batch):  # batch 是一个list，item是字典：index question target passage：title+text scores
        # assert (batch[0]['code_ids'] is not None)  # batch中的passages字段是已经拼接好了的title（函数名）和context（函数说明）的列表
        is_train = batch[0]['decoder_mask'] is not None
        index = torch.tensor([ex['index'] for ex in batch])
        if not is_train:
            target = [ex['target'] for ex in batch]
            target = self.tokenizer.batch_encode_plus(  # 批量用tokenizer将target转换成index和对应的attention_mask
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )
            target_ids = target["input_ids"]
            target_mask = target["attention_mask"].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)  # 把pad转换成-100
            target_mask = None  # 原因是为了让模型自己生成mask，跟原本的保持一样
        else:
            # TODO 直接获取code_idx，然后转换成tensor
            target_ids = [ex['code_ids'] for ex in batch]
            target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True)
            target_ids = torch.where(target_ids == 0, -100, target_ids)

            # TODO 最小化mask，将code和dataflow裁剪,code的部分和target_ids保持以一样的长度
            target_mask = None
            if is_train:
                target_mask = torch.tensor([ex['decoder_mask'] for ex in batch])
                node_lens = [item['node_len'] for item in batch]
                node_max_lens = max(node_lens)
                dataflow_lens = [item['dataflow_len'] for item in batch]
                dataflow_max_lens = max(dataflow_lens)
                assert node_max_lens == target_ids.shape[1], "长度不一致"
                # 加上dataflow的长度是因为要凑成一个正方形，因为decoder的attn_mask只能是一个正方形
                target_mask = torch.cat((target_mask[:, :node_max_lens + dataflow_max_lens, :node_max_lens] ,
                               target_mask[:, :node_max_lens + dataflow_max_lens,
                               self.code_maxlength: self.code_maxlength + dataflow_max_lens]),dim=-1)
                # target_ids同理
                pads = torch.full((target_ids.size()[0], dataflow_max_lens), self.tokenizer.pad_token_id)
                target_ids = torch.cat((target_ids, pads), dim=1)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]

        text_passages = [append_question(example) for example in batch]  # [ques + 一个passage]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)  # 两者的shape都是[4,10,100]

        return (index, target_ids, target_mask, passage_ids, passage_masks)


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:  # TODO 添加score，原因未知
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
               self.passage_prefix + " " + example[1]
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
