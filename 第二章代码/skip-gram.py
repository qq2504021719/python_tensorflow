import collections
import math
import random
import jieba
import numpy as np
import tensorflow as tf
import os


# 1.从文件中提取停止词和训练文本
def read_data():
    # 读取停用词
    stop_words = []
    with open('data/stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        # 按行读取
        line = fStopWords.readline()
        while line:
            stop_words.append((line[:-1]))  # 去除\n
            line = fStopWords.readline()
    stop_words = set(stop_words)
    print("停用词读取完毕,共{n}个词".format(n=len(stop_words)))

    # 读取文本、预处理、分词、去除停用词，得到词典
    # 得到文件夹下的文件
    s_folder_path = "data/materials"
    ls_files = []
    for root, dirs, files in os.walk(s_folder_path):
        for file in files:
            if file.endswith(".txt"):
                ls_files.append(os.path.join(root, file))

    # 读取文件的值
    raw_word_list = []
    for item in ls_files:
        with open(item, "r", encoding="UTF-8") as f:
            line = f.readline()
            while line:
                # 替换换行符
                while '\n' in line:
                    line = line.replace('\n', '')
                # 替换空格
                while ' ' in line:
                    line = line.replace(' ', '')

                # 如果句子非空
                if len(line) > 0:
                    # 调用jieba分词
                    raw_words = list(jieba.cut(line, cut_all=False))
                    for _item in raw_words:
                        # 去除停用词,_item若果不存在于raw_words，则加入raw_word_list
                        if _item not in stop_words:
                            raw_word_list.append(_item)
                line = f.readline()

    return raw_word_list


words = read_data()

print("Data size", len(words))
print("words:", words)

# 2.建立词典以及生僻词用UNK代替
vocabulary_size = 100000

def build_dataset(arg_words):
    # 词汇编码
    l_count = [['UNK', -1]]
    # 得到词出现的频次
    l_count.extend((collections.Counter(arg_words).most_common(vocabulary_size-1)))
    print("l_count", len(l_count))
    l_dictionary = dict()
    for word, _ in l_count:
        l_dictionary[word] = len(l_dictionary)

    # 使用生产的词汇编码将前面的string list[arg_words]转变为num list[data]
    l_data = []
    unk_count = 0
    for word in arg_words:
        if word in l_dictionary:
            index = l_dictionary[word]
        else:
            index = 0
            unk_count += 1
        l_data.append(index)
    l_count[0][1] = unk_count

    # 反转字典key为词汇编码，values为词汇本身
    l_reverse_dictionary = dict(zip(l_dictionary.values(), l_dictionary.keys()))
    return l_data, l_count, l_dictionary, l_reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(arg_words=words)
# print('l_data:', data)
# print('l_count:', count)
# print('l_dictionary:', dictionary)
# print('l_reverse_dictionary:', reverse_dictionary)
# 删除words节省内存
del words

data_index = 0

# 3.为skip-gram模型生产训练参数
def generate_batch(arg_batch_size, arg_num_skips, arg_skip_windows):
    global data_index

    l_batch = np.ndarray(shape=arg_batch_size, dtype=np.int32)     # (1, arg_batch_size)
    l_labels = np.ndarray(shape=(arg_batch_size, 1), dtype=np.int32)     # (arg_batch_size,1)
    span = 2 * arg_skip_windows + 1     # 示例[我 爱 祖国]
    buffer = collections.deque(maxlen=span)

    # 把词尽可能的随机分词组合
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(arg_batch_size // arg_num_skips):
        target = arg_skip_windows
        targets_to_avoid = [arg_skip_windows]

        for j in range(arg_num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            l_batch[i * arg_num_skips + j] = buffer[arg_skip_windows]
            l_labels[i * arg_num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return l_batch, l_labels


# 显示示例
batch, labels = generate_batch(arg_batch_size=8, arg_num_skips=2, arg_skip_windows=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# 4.构建模型
batch_size = 128
embedding_size = 100
skip_window = 1
num_skips = 2
valid_size = 4  # 这个数字要和len(valid_word)对应,否则保存
valid_window = 100
num_sampled = 64

# 验证集
valid_word = ['说', '王斗', '害怕', '战袄']
valid_examples = [dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # 输入数据
    train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 权重矩阵
    embeddings = tf.Variable(tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # 选取张量embeddings中对应train_inputs索引值
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 转化变量输入,适配NCE
    nce_weights = tf.Variable(
        tf.random.truncated_normal([vocabulary_size, embedding_size],
        stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         inputs=embed,
                                         labels=train_labels,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size
                        ))

    # 优化器选择
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 使用所学习的词向量来计算一个给定的minibatch与所有单词之间的相似度
    norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keepdims=True))
    normnalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normnalized_embeddings, valid_dataset)
    similary = tf.matmul(valid_embeddings, normnalized_embeddings, transpose_b=True)

    init = tf.compat.v1.global_variables_initializer()

# 训练模型
num_steps = 100001

with tf.compat.v1.Session(graph=graph) as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}

        _,loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                print("Average loss at step ", step, ":", average_loss)
                average_loss = 0

        if step % 10000 == 0:
            sim = similary.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:"% valid_word
                for k in range(top_k):
                    if k in nearest:
                        if nearest[k] in reverse_dictionary:
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s,"%(log_str, close_word)
                print(log_str)
    final_embeddings = normnalized_embeddings.eval()


# 6.输出词向量
with open('output/word2vect.txt', "w", encoding="UTF-8") as fW2V:
    fW2V.write(str(vocabulary_size) + ' ' + str(embedding_size) + '\n')
    for i in range(final_embeddings.shape[0]):
        if i in reverse_dictionary:
            sWord = reverse_dictionary[i]
            sVector = ""

            for j in range(final_embeddings.shape[1]):
                sVector = sVector + ' ' + str(final_embeddings[i, j])
            fW2V.write(sWord + sVector + '\n')