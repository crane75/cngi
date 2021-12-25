import dpkt
import numpy as np
import os
import tensorflow as tf
import sklearn
from sklearn import model_selection
import matplotlib.pyplot as plt
import json


def preProcOneFile(inputfile, padsize):
    # 打开文件
    with open(inputfile, 'rb') as pktfile:
        # 获得文件名和扩展名
        base_filename = os.path.basename(inputfile)
        (filename, extension) = os.path.splitext(base_filename)
        # 根据扩展名不同，采用不同的函数将文件读入缓存
        if extension == '.pcap':
            buff = dpkt.pcap.Reader(pktfile)
        else:
            buff = dpkt.pcapng.Reader(pktfile)

        # 初始化一个空列表，用于保存所有的报文序列
        all_pkts = []

        # 遍历整个报文，得到一个嵌套的列表
        for (ts, oneRow) in buff:
            onepkt = np.fromiter(oneRow,dtype="uint8")

            if onepkt[12] == 0x08 and onepkt[13] == 0x06:
                continue                    # if is a ARP packet, skip and continue
            if onepkt[12] == 0x08 and onepkt[13] == 0x00:   # it is an IPv4 packet
                if onepkt[23] == 0x01:
                    continue                          # if is a ICMP packet, skip and continue
                if (onepkt[34] == 0x00 and onepkt[35] == 0x35) \
                        or (onepkt[36] == 0x00 and onepkt[37] == 0x35)\
                        or (onepkt[34] == 0x00 and onepkt[35] == 0x89)\
                        or (onepkt[36] == 0x00 and onepkt[37] == 0x89)\
                        or (onepkt[34] == 0x14 and onepkt[35] == 0xEB)\
                        or (onepkt[36] == 0x14 and onepkt[37] == 0xEB)\
                        or (onepkt[34] == 0x00 and onepkt[35] == 0xA1)\
                        or (onepkt[36] == 0x00 and onepkt[37] == 0xA1)\
                        or (onepkt[34] == 0x00 and onepkt[35] == 0x7b)\
                        or (onepkt[36] == 0x00 and onepkt[37] == 0x7b)\
                        or (onepkt[34] == 0x2b and onepkt[35] == 0x69)\
                        or (onepkt[36] == 0x2b and onepkt[37] == 0x69)\
                        or (onepkt[34] == 0x32 and onepkt[35] == 0xc8)\
                        or (onepkt[36] == 0x32 and onepkt[37] == 0xc8)\
                        or (onepkt[34] == 0x00 and onepkt[35] == 0x50)\
                        or (onepkt[36] == 0x00 and onepkt[37] == 0x50):
                    continue         # if is a DNS , NBNS, LLMNR ,SNMP, ntp, 80 packet then skip and continue
#            else:
#                if onepkt[12] == 0x86 and onepkt[13] == 0xdd:   # it is an IPv6 packet
            onepkt = set_mac_ip(onepkt)
            all_pkts.append(onepkt.tolist())
        # 填充random int between 0 and 255 或者截取长度
        output_array = tf.keras.preprocessing.sequence.pad_sequences(
            all_pkts, maxlen=padsize, dtype='uint8', padding='post', truncating='post',
            value=0)
        # 生成分类数组,数值相同,为文件名前5个字符
        pktfile.close()
        return output_array


# MAC地址和IP地址置0
def set_mac_ip(pkt):
    if pkt[12] == 0x08 and pkt[13] == 0x00:  # it is an IPv4 packet
        for i in range(12):
            pkt[i] = 0
        for j in range(26,34):
            pkt[j] = 0
    else:
        if pkt[12] == 0x86 and pkt[13] == 0xdd:  # it is an IPv6 packet
            for i in range(12):
                pkt[i] = 0
            for j in range(22, 54):
                pkt[j] = 0
    return pkt


# 生成分类名及索引保存在json文件
def set_cat_json(dirname):
    filenames = []
    filelist = os.listdir(dirname)
    for fname in filelist:
        (filename, extension) = os.path.splitext(fname)
        filenames.append(filename[:5])
    dic = dict(enumerate(set(filenames)))
    dict_cat = dict(zip(dic.values(), dic.keys()))
    try:
        with open("cnn/category.json", 'w') as cat_jsonfile:
            json.dump(dict_cat,cat_jsonfile)
    except IOError as e:
        print(e)
    cat_jsonfile.close()
    return dict_cat


# 从json文件中读取分类字典
def get_cat_from_json():
    cat = []
    try:
        with open( "cnn/category.json", 'r') as cat_jsonfile:
            cat = json.load(cat_jsonfile)
    except IOError as e:
        print(e)
    cat_jsonfile.close()
    return cat


def preProcessDir(DIR):
    pktsize = 900
    x = np.zeros([1, pktsize], dtype='uint8')
    y = np.zeros([1, 1], dtype='uint8')
    allCategory = get_cat_from_json()
    #  便利整个目录，得到全部数据
    for fname in os.listdir(DIR):
        cat_name = fname[:5]
        x_oneFile = preProcOneFile(os.getcwd() + '/' + DIR + '/' + fname, pktsize)
        y_oneFile = np.full([1, len(x_oneFile)], allCategory[cat_name]).transpose()
        x = np.concatenate((x, x_oneFile))
        y = np.concatenate((y, y_oneFile))

    # 删除初始化时多增加的一行全0元素
    x = np.delete(x, [0], axis=0)
    x = x.reshape([len(x), 30, 30, 1])
    y = np.delete(y, [0], axis=0)
    return x/256, y


# 本程序已经完成，科研得到precision,recall,f1_score,，另作一个进行验收

def plot_loss(history):
    plt.figure()
    #plt.title('训练次数-损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.xticks(history.epoch)
    plt.plot(history.epoch,history.history['loss'])
    plt.show()
    plt.savefig('cnn/static/cnn/loss.png')


def plot_accuracy(history):
    plt.figure()
    #plt.title('训练次数-精确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.xticks(history.epoch)
    plt.plot(history.epoch,history.history['accuracy'])
    plt.show()
    plt.savefig('cnn/static/cnn/accuracy.png')


def train(trainsize=0.8, batchsize=512, stride=1, learn_rate=0.01, epoch=10):
    dir = "cnn/static/datas"     # 训练数据所在目录
    cat = set_cat_json(dir)
    catnumber = len(cat)      # 分类的个数
    X, Y = preProcessDir(dir)
    Y = tf.keras.utils.to_categorical(Y, catnumber)
    X_train, X_test, Y_train,  Y_test = \
        model_selection.train_test_split(X, Y, train_size=trainsize, random_state=14)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', strides=stride, input_shape=[30,30,1]),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(catnumber, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batchsize, epochs=epoch, verbose=0)
    plot_loss(model.history)
    plot_accuracy(model.history)
    Y_predict = tf.keras.utils.to_categorical(
        model.predict_classes(X_test, batch_size=512, verbose=0), catnumber)
    target_name = get_cat_from_json()

    report = sklearn.metrics.classification_report(
        Y_test, Y_predict, target_names=target_name, sample_weight=None, output_dict=True)
    model.save('cnn/my_model')
    return report


def classfy():
    # 生成分类名称与分类索引的字典
    dict_classes = get_cat_from_json()

    # 执行分类操作
    model = tf.keras.models.load_model('cnn/my_model')
    classy_data_path = os.getcwd() + '/cnn/classify-data'
    files = os.listdir(classy_data_path)
    padsize = 900
    # 初始化一个空列表，用于保存所有的报文序列
    all_pkts = []

    for file in files:
        classify_path_file = classy_data_path + '/' + file
        with open(classify_path_file,'rb') as pktfile:
            basefilename = os.path.basename(classify_path_file)
            (filename, extension) = os.path.splitext(basefilename)
            # 根据扩展名不同，采用不同的函数将文件读入缓存
            if extension == '.pcap':
                buff = dpkt.pcap.Reader(pktfile)
            else:
                buff = dpkt.pcapng.Reader(pktfile)
            # 遍历整个报文，得到一个嵌套的列表
            for (ts, oneRow) in buff:
                onepkt = np.fromiter(oneRow, dtype="uint8")

                if onepkt[12] == 0x08 and onepkt[13] == 0x06:
                    continue  # if is a ARP packet, skip and continue
                if onepkt[12] == 0x08 and onepkt[13] == 0x00:  # it is an IPv4 packet
                    if onepkt[23] == 0x01:
                        continue  # if is a ICMP packet, skip and continue
                    if (onepkt[34] == 0x00 and onepkt[35] == 0x35) \
                            or (onepkt[36] == 0x00 and onepkt[37] == 0x35) \
                            or (onepkt[34] == 0x00 and onepkt[35] == 0x89) \
                            or (onepkt[36] == 0x00 and onepkt[37] == 0x89) \
                            or (onepkt[34] == 0x14 and onepkt[35] == 0xEB) \
                            or (onepkt[36] == 0x14 and onepkt[37] == 0xEB) \
                            or (onepkt[34] == 0x00 and onepkt[35] == 0xA1) \
                            or (onepkt[36] == 0x00 and onepkt[37] == 0xA1) \
                            or (onepkt[34] == 0x00 and onepkt[35] == 0x7b) \
                            or (onepkt[36] == 0x00 and onepkt[37] == 0x7b) \
                            or (onepkt[34] == 0x2b and onepkt[35] == 0x69) \
                            or (onepkt[36] == 0x2b and onepkt[37] == 0x69) \
                            or (onepkt[34] == 0x32 and onepkt[35] == 0xc8) \
                            or (onepkt[36] == 0x32 and onepkt[37] == 0xc8) \
                            or (onepkt[34] == 0x00 and onepkt[35] == 0x50) \
                            or (onepkt[36] == 0x00 and onepkt[37] == 0x50):
                        continue  # if is a DNS , NBNS, LLMNR ,SNMP, ntp, 80 packet then skip and continue
                #            else:
                #                if onepkt[12] == 0x86 and onepkt[13] == 0xdd:   # it is an IPv6 packet
                onepkt = set_mac_ip(onepkt)
                all_pkts.append(onepkt.tolist())
            # 填充random int between 0 and 255 或者截取长度
        pktfile.close()     # 处理完一个文件，关闭它

    output_array = tf.keras.preprocessing.sequence.pad_sequences(
        all_pkts, maxlen=padsize, dtype='uint8', padding='post',
        truncating='post', value=0)
    output_array = output_array.reshape(output_array.shape[0], 30, 30, 1)/256
    classify_result = model.predict_classes(output_array,)      # 每个报文对应的分类索引
    classify_result_statistic = np_groupby(classify_result)     # 统计每个分类索引的综述

    #print('groupby result is :' + str(classify_result_statistic))
    #print('所有分类索引及名称是' + str(dict_classes))
    report = dict()
    #print(classify_result_statistic)
    target_name_reverse = dict(zip(dict_classes.values(), dict_classes.keys()))
    for key, value in classify_result_statistic.items():
        report[target_name_reverse[key]] = value
    return report


# 定义函数，计算numpy数组中重复数据的个数，类似groupby功能
def np_groupby(array):
    gpb = dict()
    for i in np.unique(array):
        gpb[i] = 0      # 初始化统计数
    for j in array:
        gpb[j] = gpb[j] + 1
    return gpb


def remove_train_file(filename):
    DIR = "cnn/static/datas"
    filename = DIR + '/' + filename
    os.remove(filename)


def remove_classify_file(filename):
    DIR = "cnn/classify-data"
    filename = DIR + '/' + filename
    os.remove(filename)



if __name__ == '__main__':
    main()

