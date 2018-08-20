# coding: utf-8
import os
import sys
import time


LETTER_NUM = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
BIG_LETTER_NUM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
NUM_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 棋盘字母位置速查表
seq_lookup = dict(zip(LETTER_NUM, NUM_LIST))
num2char_lookup = dict(zip(NUM_LIST, BIG_LETTER_NUM))

# SGF文件
sgf_home = './sgf_data/'
width = 15


def get_files_as_list(data_dir):
    # 扫描某目录下SGF文件列表
    file_list = os.listdir(data_dir)
    file_list = [item for item in file_list if item.endswith('.sgf') and os.path.isfile(os.path.join(data_dir, item))]
    return file_list


def content_to_order(sequence):
    # 棋谱字母转整型数字

    global seq_lookup   # 棋盘字母位置速查表
    seq_list = sequence.split(';')
    # list:['hh', 'ii', 'hi'....]
    seq_list = [item[2:4] for item in seq_list]
    # list: [112, 128, ...]
    seq_num_list = [seq_lookup[item[0]]*width+seq_lookup[item[1]] for item in seq_list]
    return seq_list, seq_num_list


def num2char(order_):
    global num2char_lookup
    Y_axis = num2char_lookup[order_/width]
    X_axis = num2char_lookup[order_ % width]
    return '%s%s' % (Y_axis, X_axis)


def read_one_file(file_name, data_dir):
    with open(os.path.join(data_dir, file_name),encoding="GBK" ) as f:
    # with open(os.path.join(data_dir, file_name)) as f:
        p = f.read()
        start = file_name.index('_') + 1
        end = file_name.index('_.')

        inxstr = "SZ[{}]".format(width)
        sequence = p[p.index(inxstr)+7:-4]
        try:
            seq_list, seq_num_list = content_to_order(sequence)
        except Exception as e:
            print('***' * 20 + "\n e is {} and file name is {}".format(e,file_name))

        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'Blank' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'blank':
            winner = 1
        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'White' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'white':
            winner = 2
        return {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'file_name':file_name}


def get_data_from_files(file_name, data_dir):
    assert file_name.endswith('.sgf'), 'file: %s 不是SGF文件' % file_name
    with open(os.path.join(data_dir, file_name)) as f:
        p = f.read()
        # 棋谱内容 开始/结束 位置
        start = file_name.index('_') + 1
        end = file_name.index('_.')

        sequence = p[p.index('SZ[width]')+7:-4]
        try:
            seq_list, seq_num_list = content_to_order(sequence)
        except Exception as e:
            print('***' * 20)
            print(e)
            print(file_name)
        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'Blank' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'blank':
            winner = 1 
        if file_name[file_name.index('_')+1:file_name.index('_')+6] == 'White' or file_name[file_name.index('_')+1:file_name.index('_')+6] == 'white':
            winner = 2
        return {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'file_name':file_name}


def read_files(data_dir):
    file_list = get_files_as_list(data_dir)
    index = 0
    while True:
        if index >= len(file_list): yield None
        with open(data_dir+file_list[index]) as f:
            p = f.read()
            # 棋谱内容 开始/结束 位置
            start = file_list[index].index('_') + 1
            end = file_list[index].index('_.')

            sequence = p[p.index('SZ[width]')+7:-4]
            try:
                seq_list, seq_num_list = content_to_order(sequence)
            except Exception as e:
                print('***' * 20)
                print(e)
                print(file_list[index])
        if sequence[-5] == 'B' or sequence[-5] == 'b':
            winner = 1 
        if sequence[-5] == 'W' or sequence[-5] == 'w':
            winner = 2
        yield {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'index': index, 'file_name':file_list[index]}
        index += 1

def num2char(order_):
        global num2char_lookup
        Y_axis = num2char_lookup[order_/width]
        X_axis = num2char_lookup[order_ % width]
        return '%s%s' % (Y_axis, X_axis)


if __name__ == '__main__':
    # data = read_files(sgf_home)
    # print("\n".join(get_files_as_list(sgf_home)))
    contents = []
    files = get_files_as_list(sgf_home)
    for one_file in files :
        content = read_one_file(one_file,sgf_home)
        contents.append(content)

    print(contents[100])

    import pickle

    with open('entry.pickle', 'wb') as fw:
        pickle.dump(contents, fw)

    with open('entry.pickle', 'rb') as fr:
        content_ld = pickle.load(fr)

    print(content_ld[100])
    print(content_ld[99])