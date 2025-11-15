import os
import shutil
import hashlib
from sys import platform
from gensim.models import KeyedVectors

import os

print("当前工作目录:", os.getcwd())
print("脚本所在目录:", os.path.dirname(os.path.abspath(__file__)))
print("文件是否存在:", os.path.exists("glove.840B.300d.txt"))

# 列出当前目录所有文件
print("当前目录文件:")
for file in os.listdir('.'):
    print(" -", file)
def prepend_line(infile, outfile, line):
    """
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r', encoding='utf-8') as old:
        with open(outfile, 'w', encoding='utf-8') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r', encoding='utf-8') as fin:
        with open(outfile, 'w', encoding='utf-8') as fout:
            fout.write(line + "\n")
            for line_content in fin:
                fout.write(line_content)


def checksum(filename):
    """
    This is to verify the file checksum is the same as the glove files we use to
    pre-computed the no. of lines in the glove file(s).
    """
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


# Pre-computed glove files values.
pretrain_num_lines = {
    "glove.840B.300d.txt": 2196017,
    "glove.42B.300d.txt": 1917494,
    "glove.6B.300d.txt": 400000,
    "glove.6B.200d.txt": 400000,
    "glove.6B.100d.txt": 400000,
    "glove.6B.50d.txt": 400000
}

pretrain_checksum = {
    "glove.6B.300d.txt": "b78f53fb56ec1ce9edc367d2e6186ba4",
    "glove.twitter.27B.50d.txt": "6e8369db39aa3ea5f7cf06c1f3745b06",
    "glove.42B.300d.txt": "01fcdb413b93691a7a26180525a12d6e",
    "glove.6B.50d.txt": "0fac3659c38a4c0e9432fe603de60b12",
    "glove.6B.100d.txt": "dd7f3ad906768166883176d69cc028de",
    "glove.twitter.27B.25d.txt": "f38598c6654cba5e6d0cef9bb833bdb1",
    "glove.6B.200d.txt": "49fa83e4a287c42c6921f296a458eb80",
    "glove.840B.300d.txt": "eec7d467bccfa914726b51aac484d43a",
    "glove.twitter.27B.100d.txt": "ccbdddec6b9610196dd2e187635fee63",
    "glove.twitter.27B.200d.txt": "e44cdc3e10806b5137055eeb08850569",
}


def check_num_lines_in_glove(filename, check_checksum=False):
    """
    根据文件名返回GloVe文件的词汇表大小
    """
    # 只使用文件名，不包含路径
    base_filename = os.path.basename(filename)

    if check_checksum:
        assert checksum(filename) == pretrain_checksum[base_filename]

    if base_filename.startswith('glove.6B.'):
        return 400000
    elif base_filename.startswith('glove.twitter.27B.'):
        return 1193514
    else:
        return pretrain_num_lines.get(base_filename, 0)


def count_lines_manual(filename):
    """
    手动计算文件行数（如果预定义的值不可用）
    """
    print(f"正在手动计算文件行数: {filename}")
    count = 0
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for count, _ in enumerate(f, 1):
            pass
    return count


def get_dimensions(filename):
    """
    从文件第一行获取向量维度
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        # 第一行的格式：word dim1 dim2 dim3 ...
        dimensions = len(first_line.split()) - 1
    return dimensions


def main():
    # 使用原始字符串（在字符串前加r）来避免转义序列警告
    glove_file = r"glove.840B.300d.txt"  # 使用原始字符串

    # 检查文件是否存在
    if not os.path.exists(glove_file):
        print(f"错误：找不到GloVe文件: {glove_file}")
        print("请确保：")
        print("1. 文件存在于当前目录")
        print("2. 文件名拼写正确")
        print("3. 或者提供完整的文件路径")
        return

    print(f"找到GloVe文件: {glove_file}")
    print(f"文件大小: {os.path.getsize(glove_file) / (1024 ** 3):.2f} GB")

    # 获取词汇表大小和维度
    try:
        num_lines = check_num_lines_in_glove(glove_file)
        if num_lines == 0:
            num_lines = count_lines_manual(glove_file)
    except KeyError:
        print("无法从预定义数据获取行数，正在手动计算...")
        num_lines = count_lines_manual(glove_file)

    dimensions = get_dimensions(glove_file)

    print(f"词汇表大小: {num_lines}")
    print(f"向量维度: {dimensions}")

    # 输出文件
    gensim_file = 'glove_model.txt'
    gensim_first_line = f"{num_lines} {dimensions}"

    print(f"正在转换文件，输出到: {gensim_file}")

    # 根据平台选择方法
    if platform == "linux" or platform == "linux2":
        prepend_line(glove_file, gensim_file, gensim_first_line)
    else:
        prepend_slow(glove_file, gensim_file, gensim_first_line)

    print("文件转换完成!")

    # 验证转换结果
    try:
        print("正在验证转换结果...")
        # 使用现代API加载
        model = KeyedVectors.load_word2vec_format(gensim_file, binary=False, no_header=True)

        print(f"模型加载成功!")
        print(f"词汇表大小: {len(model.key_to_index)}")
        print(f"向量维度: {model.vector_size}")

        # 测试一些功能
        test_words = ['the', 'and', 'is', 'in', 'to']
        found_words = [word for word in test_words if word in model]
        print(f"测试词汇找到 {len(found_words)}/{len(test_words)} 个常用词")

        if 'king' in model and 'queen' in model:
            similarity = model.similarity('king', 'queen')
            print(f"'king'和'queen'的相似度: {similarity:.4f}")

        if 'australia' in model:
            similar_words = model.most_similar(positive=['australia'], topn=5)
            print("与'australia'最相似的词:")
            for word, score in similar_words:
                print(f"  {word}: {score:.4f}")

    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        print("但文件转换可能已经成功，你可以尝试手动加载")


if __name__ == "__main__":
    main()