import tensorflow as tf
import numpy as np
import helper

token_dictm, vocab_to_int, int_to_vocab = helper.load_data()
seq_length, load_dir = helper.load_params()


# 获取缓存的tensor
def get_tensors(arg_loaded_graph):
    inputs = arg_loaded_graph.get_tensor_by_name("inputs:0")
    l_initial_state = arg_loaded_graph.get_tensor_by_name("init_state:0")
    l_final_state = arg_loaded_graph.get_tensor_by_name("final_state:0")
    l_probes = arg_loaded_graph.get_tensor_by_name("probes:0")
    return inputs, l_initial_state, l_final_state, l_probes

# 从字典里面随机取出词汇
def pick_word(arg_probabilities, arg_int_to_vacab):
    num_word = np.random.choice(len(arg_int_to_vacab), p=arg_probabilities)

    return arg_int_to_vacab[num_word]


# 用生成的模型生成小说内容
# 生成的文本长度
gen_lenth = 500

# 文章开头的字,指定一个就好,但是这个字必须在训练词汇表里
prime_word = "州"

loaded_graph = tf.Graph()
with tf.compat.v1.Session(graph=loaded_graph) as sess:
    # 加载保存过的session
    loader = tf.compat.v1.train.import_meta_graph(load_dir + ".meta")
    loader.restore(sess, load_dir)

    # 通过名称获取缓存的tensor
    input_text, initial_state, final_state, probes = get_tensors(loaded_graph)

    # 准备开始生成文本
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, feed_dict={input_text:np.array([[1]])})

    # 开始生成文本
    for n in range(gen_lenth):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_lenth = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probes, final_state],
            feed_dict={input_text: dyn_input, initial_state:prev_state}
        )

        probes_array = probabilities[0][dyn_seq_lenth - 1]
        pred_word = pick_word(probes_array, int_to_vocab)
        gen_sentences.append(pred_word)

    # 将标点符号还原
    novel = ''.join(gen_sentences)
    for key, token in token_dictm.items():
        ending = '' if key in ['\n', '(', '"'] else ''
        novel = novel.replace(token, key)

    print(novel)
