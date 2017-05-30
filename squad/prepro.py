import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(asctime)s:%(message)s'
)

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    # 这个action只有在存在某个参数命令的时候，如--xxx，才会自己设置默认值，否则为False
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train-v1.1.json')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--suffix", default="")
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, args.train_name)
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, args.dev_name)
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    '''
    将处理后的数据存储成json
    :param args: 
    :param data: 
    :param shared: 
    :param data_type: 
    :return: 
    '''
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    logging.info('data:{}'.format(data_path))
    logging.info('shared:{}'.format(shared_path))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    '''
    将word_counter中存在的词转换了glove词嵌入，如果glove词嵌入中不存在某个词，那么不做处理
    :param args: 
    :param word_counter:Counter对象 
    :return: 字典{词:词向量的list}
    '''
    # 默认是100维的词向量
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    # 文件的行数
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    # 文件的格式如下
    # example 0d 1d 2d ...
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            # 起始位置是词
            word = array[0]
            # 剩下的位置的数据是词嵌入
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            # 将字符串的第一个字符转换为大写
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    '''
    将squad数据集train或者dev处理成模型期待的格式
    squad的数据结构是如下组织的：
    {
        'data':
            [
                {（某一篇文章）
                    'paragraphs':
                        [
                            {（文章中的某个段落）
                                'context':
                                'qas':
                                    [
                                        {（针对这个段落提出的某个问题）
                                            'id':
                                            'question':                                    
                                            'answers':
                                                [（针对某个问题一共有3个备选答案）
                                                    {（针对某个问题给出的备选答案）
                                                        'answer_start':（从段落中的第几个字符开始是答案）
                                                        'text':
                                                    }
                                                ]
                                        },...
                                    ]
                            },...
                        ]
                    'title':
                },...
            ]
        'version':
    }
    :param args: 
    :param data_type: 
    :param start_ratio: 
    :param stop_ratio: 
    :param out_name: 
    :param in_path: 
    :return: 
    '''
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    # 如果不进行分句
    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-{}v1.1.json".format(data_type, args.suffix))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    na = []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    # 第几篇passage开始
    start_ai = int(round(len(source_data['data']) * start_ratio))
    # 第几篇passage结束
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    # 对每一篇文章
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        # xp存储对应于每一个段落的xi
        # cxp存储对应于每一个段落的cxi
        xp, cxp = [], []
        pp = []
        # x存储对应于每一篇文章的xp
        # cx存储对应于每一篇文章的cxp
        # p存储对应于每一篇文章的pp
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        # 对每一个段落
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            # 首先将context分句，然后对每一句分词，xi的结构如下：
            # [（每一句）
            #   [（每一句中的每一个词）
            #       xxx,xxx,xxx,xxx
            #   ],...
            # ]
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            # xij是每一句话中的词的list
            # xijk是指的每一个词
            # list(xijk)将str转换成了list，它是一个字符的list，每个list表示一个词
            # cxi和xi结构类似，只是每一个词用list表示而不是str表示
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            # pp存储对应每一篇文章的原始的context
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    # xijk是指的每一个词
                    # len(para['qas']表示的是一个段落有多少个问题
                    # todo:不知道这里的计数是什么意思
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            # 对于每个问题
            for qa in para['qas']:
                # get words
                # qi表示问题中各个词组成的list
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                # 和qi类似，每个词用str表示
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                # 存储问题的答案的list，每个元素都是备选答案
                answers = []
                # 对于某个问题的每一个答案
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    # yi0表示答案起始位置的词的索引
                    # yi1表示答案终止位置的词的索引+1
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    # w0答案起始位置的词
                    w0 = xi[yi0[0]][yi0[1]]
                    # w1答案终止位置的词
                    w1 = xi[yi1[0]][yi1[1]-1]
                    # 起始位置的词的开始位置在context中的索引
                    i0 = get_word_idx(context, xi, yi0)
                    # 终止位置的词的开始位置在context中的索引
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    # 真实答案相对于起始位置词的偏移量，可以看做是索引
                    cyi0 = answer_start - i0
                    # 真实答案相对于终止位置词的偏移量，可以看做是索引
                    # 多减一个1是应为answer_stop指向的位置是真实答案结束位置的索引加1
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                # 检查某个问题是不是没有答案
                if len(qa['answers']) == 0:
                    yi.append([(0, 0), (0, 1)])
                    cyi.append([0, 1])
                    na.append(True)
                else:
                    na.append(False)
                # todo:不知道这里的计数是什么意思
                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                # [（每个问题）
                #   [（问题中的各个词的str）]
                # ]
                q.append(qi)
                # [（每个问题）
                #   [（问题中的各个词的list）]
                # ]
                cq.append(cqi)
                # [（每个问题）
                #   [（每个备选答案）
                #       [（句号，词号），（句号，词号）]
                #   ]
                # ]
                y.append(yi)
                # [（每个问题）
                #   [（每个备选答案）
                #       [
                #           真实答案起始位置对应的起始词内索引,
                #           真实答案结束位置对应的起始词内索引
                #       ]
                #   ]
                # ]
                cy.append(cyi)
                # [（每个问题）
                #   [文章号索引，段落号索引]
                # ]
                rx.append(rxi)
                # [（每个问题）
                #   [文章号索引，段落号索引]
                # ]
                rcx.append(rxi)
                # [（每个问题）
                #   问题的id号
                # ]
                ids.append(qa['id'])
                # [（每个问题）
                #   处理的第几个问题的索引
                # ]
                idxs.append(len(idxs))
                # [（每个问题）
                #   [（每个备选答案）
                #       真实答案原文
                #   ]
                # ]
                answerss.append(answers)

        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    # q:
    # [（每个问题）
    #   [（问题中的各个词的str）]
    # ]

    # cq:
    # [（每个问题）
    #   [（问题中的各个词的list）]
    # ]

    # y:
    # [（每个问题）
    #   [（每个备选答案）
    #       [（句号，词号），（句号，词号）]
    #   ]
    # ]

    # *x:
    # [（每个问题）
    #   [文章号索引，段落号索引]
    # ]

    # *cx:
    # [（每个问题）
    #   [文章号索引，段落号索引]
    # ]

    # cy:
    # [（每个问题）
    #   [（每个备选答案）
    #       [
    #           真实答案起始位置对应的起始词内索引,
    #           真实答案结束位置对应的起始词内索引
    #       ]
    #   ]
    # ]

    # idxs:
    # [（每个问题）
    #   处理的第几个问题的索引
    # ]

    # ids:
    # [（每个问题）
    #   问题的id号
    # ]

    # answerss:
    # [（每个问题）
    #   [（每个备选答案）
    #       真实答案原文
    #   ]
    # ]

    # *p:
    # [（每个问题）
    #   [文章号索引，段落号索引]
    # ]

    # na:
    # [（每个问题）
    #   问题的是否存在答案（True，False）
    # ]

    # 其中带*的键名都是和某个文章，某个段落有关的键，有x，cx，p，它存储的都是文章对应内容的索引
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}
    # x:
    # [（每一篇文章）
    # 	[（每一个段落）
    # 		[（每一句）
    # 		  [（每一句中的每一个词用str表示）
    # 		      xxx,xxx,xxx,xxx
    # 		  ],...
    # 		]
    # 	]
    # ]

    # cx:
    # [（每一篇文章）
    # 	[（每一个段落）
    # 		[（每一句）
    # 		  [（每一句中的每一个词用list表示）
    # 		      xxx,xxx,xxx,xxx
    # 		  ],...
    # 		]
    # 	]
    # ]

    # p:
    # [（每一篇文章）
    # 	[（每一个段落）
    # 		每一个段落的原始文本str
    # 	]
    # ]

    # word_counter:
    # {（所有的(段落，问题)数据对中，某个词的出现次数）
    # 	str:int
    # }

    # char_counter:
    # {（所有的(段落，问题)数据对中，某个字母的出现次数）
    # 	str:int
    # }

    # lower_word_counter:
    # {（所有的(段落，问题)数据对中，某个词的小写形式的出现次数）
    # 	str:int
    # }

    # word2vec:
    # {
    #   词:词向量的list（str:list）
    # }

    # lower_word2vec:
    # {
    #   词:词的小写形式的词向量的list（str:list）
    # }
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)

if __name__ == "__main__":
    main()