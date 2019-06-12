import numpy as np
import keras as kr

def open_file(filename, mode='r'):
    return open(filename,encoding="UTF-8")

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels

# words,word_to_id,classList,cat_to_id
def get_words_cat_dir():
    words = []
    with open('data/cnews.vocab.txt', encoding="UTF-8") as f:
        words.extend(f.readlines())
        words = [i.strip() for i in words]
        print(words)
        print(len(words))

    word_to_id = dict(zip(words, range(len(words))))
    print(word_to_id)
    classList = ['phone', 'weather', 'translation', 'playcontrol', 'volume', 'FM',
                 'limitLine', 'alarm', 'schedule', 'music', 'story',
                 'news', 'collect', 'musicinfo', 'healthAI', 'calculator', 'cookbook',
                 'dictionary', 'joke', 'forex', 'stock', 'other']
    cat_to_id = dict(zip(classList, range(len(classList))))
    print(cat_to_id)
    return words,word_to_id,classList,cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=60):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length,truncating='post',padding='post')
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def process_one_strLine(strLine,word_to_id, max_length=60):
    data_id = [[word_to_id[x] for x in strLine if x in word_to_id]]
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, truncating='post', padding='post')
    return x_pad

def process_list_strLine(strLine,word_to_id, max_length=60):
    result = []
    for one in strLine:
        result.append([word_to_id[x] for x in one if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(result, max_length, truncating='post', padding='post')
    return x_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]