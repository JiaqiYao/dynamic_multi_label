import os
import json
from tqdm import tqdm


def build_vocabulary(data_dir):
    with open(os.path.join(data_dir, 'train_texts.txt'), 'rt') as fin:
        train_texts = json.load(fin)
    print("train text cuts load done")
    with open(os.path.join(data_dir, "Telegram", 'train_key_words.dat'), 'rb') as fin:
        train_key_words = pickle.load(fin)
    print("train key_words load done")
    words = set()
    for train_text in tqdm(train_texts,miniters=1000):
        for word in train_text:
            words.add(word)
    for key_word in tqdm(train_key_words,miniters=1000):
        for word in key_word:
            words.add(word)
    with open(os.path.join(data_dir, "Telegram", "words.dat"), 'wb') as fout:
        pickle.dump(words, fout)
    print("Build Vocabulary Done!!!")


def get_word_embedding(data_home, word2vec_name):
    with open(os.path.join(data_home, "Telegram", "words.dat"), 'rb') as fin:
        words = pickle.load(fin)
    telegram_word_embeddings = dict()
    print("The number of words is {}".format(len(words)))
    word2vec_path = os.path.join(data_home, "word_embedding", word2vec_name)
    with open(word2vec_path, 'rt') as fin:
        line = fin.readline()
        words_num, embed_size = line.split()
        print("The number of words is {}, the embedding size is {}".format(words_num, embed_size))
        for line in tqdm(fin, miniters=5000):
            word, embed = line.split(maxsplit=1)
            if word in words:
                try:
                    telegram_word_embeddings[word] = [float(vec) for vec in embed.split()]
                except Exception as e:
                    print(e)
                    print(line)
    vocab_size = len(telegram_word_embeddings)
    with open(os.path.join(data_home, "word_embedding", "telegram_word_embedding.dat"), 'wb') as fout:
        pickle.dump(telegram_word_embeddings, fout)
    print("done!!!")


if __name__ == "__main__":
    data_dir = r'/home/yaojq/data/text/reuters'
    word2vec_path = "/home/yaojq/data/word_embedding/GoogleNews-vectors-negative300.bin"
    print("build vocabulary")
    build_vocabulary(data_dir)
    get_word_embedding(data_dir, word2vec_path)

