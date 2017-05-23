import re
import numpy as np


def get_2d_spans(text, tokenss):
    '''
    返回tokenss中每个词在text中出现的位置
    :param text: str表示的原文本
    :param tokenss: 分句和分词的嵌套list
    :return: list(list(tuple))
    [（每一个分句）
        [（分句里每一个词）
            (起始位置，结束位置)
        ]
    ]
    '''
    spanss = []
    cur_idx = 0
    # 对于每一个分句
    for tokens in tokenss:
        spans = []
        # 对于每个词
        for token in tokens:
            # 在text中没有发现某个token
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    '''
    在context中寻找start,stop指示的范围中，wordss中哪些词被包含在内
    :param context: 一个字符串 
    :param wordss: 分句和分词的嵌套list
    :param start: 起始的index
    :param stop: 终止的index
    :return:被包含词在wordss中的索引，左闭右开，是一个切片 
    ((句号，起始词索引)，(句号，终止词索引+1))
    '''
    spanss = get_2d_spans(context, wordss)
    idxs = []
    # 对每一个分句
    for sent_idx, spans in enumerate(spanss):
        # 每一个分句里的每一个词
        for word_idx, span in enumerate(spans):
            # 如果span和(start,stop)有所重叠，也就是说某个词在答案的范围之内
            # 将某个词所属的句子的index和词的index翻入idxs中
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def get_phrase(context, wordss, span):
    """
    Obtain phrase as substring of context given start and stop indices in word level
    :param context:
    :param wordss:
    :param start: [sent_idx, word_idx]
    :param stop: [sent_idx, word_idx]
    :return:
    """
    start, stop = span
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]


def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]


def process_tokens(temp_tokens):
    '''
    针对一些特殊符号，将temp_tokens中没有分开的词组进一步分隔开
    :param temp_tokens: [xxx,...]
    :return: 
    '''
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        # 正则表达式：'([-−—–/~"\'“’”‘°])'，使用上面的正则表达式进行分割
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def get_best_span(ypi, yp2i):
    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        for j in range(len(ypif)):
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 * val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 * val2
    return ((best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)), float(max_val)


def get_best_span_wy(wypi, th):
    chunk_spans = []
    scores = []
    chunk_start = None
    score = 0
    l = 0
    th = min(th, np.max(wypi))
    for f, wypif in enumerate(wypi):
        for j, wypifj in enumerate(wypif):
            if wypifj >= th:
                if chunk_start is None:
                    chunk_start = f, j
                score += wypifj
                l += 1
            else:
                if chunk_start is not None:
                    chunk_stop = f, j
                    chunk_spans.append((chunk_start, chunk_stop))
                    scores.append(score/l)
                    score = 0
                    l = 0
                    chunk_start = None
        if chunk_start is not None:
            chunk_stop = f, j+1
            chunk_spans.append((chunk_start, chunk_stop))
            scores.append(score/l)
            score = 0
            l = 0
            chunk_start = None

    return max(zip(chunk_spans, scores), key=lambda pair: pair[1])


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs


