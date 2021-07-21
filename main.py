import pandas as pd


def load_dict():
    """
    加载同义词典
    :return:
    """
    synonym_dict = {}
    # df = pd.read_excel('data/synonym_dict/synonym_dict.xlsx', sheet_name=None)  # 正向最大匹配
    df = pd.read_excel('data/synonym_dict/synonym_dict_reverse.xlsx', sheet_name=None)  # 逆向最大匹配

    for key, df_tmp in df.items():
        synonym_dict[key] = dict(zip(df_tmp['标准词'], df_tmp['同义词']))

    lens = []
    for v in synonym_dict.values():
        for k in v.keys():
            lens.append(len(k))
    word_max_len = max(lens)

    return synonym_dict['n_crop'], synonym_dict['n_disease'], synonym_dict['n_medicine'], word_max_len


class NerBaseRule:
    def __init__(self, crop_dict, disease_dict, medicine_dict, word_max_len, reverse=True):
        """

        :param crop_dict: 同义词典（代表不同的 NER type）
        :param disease_dict: 同义词典（代表不同的 NER type）
        :param medicine_dict: 同义词典（代表不同的 NER type）
        :param word_max_len: 匹配前候选词长度
        :param reverse: 是否采用逆向最大匹配
        """
        self.crop_dict = crop_dict
        self.disease_dict = disease_dict
        self.medicine_dict = medicine_dict
        self.word_max_len = word_max_len
        self.reverse = reverse

    def ner(self, line):
        if self.reverse:
            line = line[::-1]

        n_crop = []
        n_disease = []
        n_medicine = []

        line_len = len(line)
        index = 0
        while line_len > 0:
            try_word = line[:self.word_max_len]
            while try_word not in self.crop_dict and try_word not in self.disease_dict and try_word not in self.medicine_dict:
                if len(try_word) == 1:
                    break
                try_word = try_word[:len(try_word) - 1]
            if try_word in self.crop_dict:
                n_crop.append(try_word)
            if try_word in self.disease_dict:
                n_disease.append(try_word)
            if try_word in self.medicine_dict:
                n_medicine.append(try_word)
            line = line[len(try_word):]
            index += len(try_word)
            line_len = len(line)

        if self.reverse:
            n_crop = list(map(lambda x: x[::-1], n_crop))
            n_disease = list(map(lambda x: x[::-1], n_disease))
            n_medicine = list(map(lambda x: x[::-1], n_medicine))
            n_crop.reverse()
            n_disease.reverse()
            n_medicine.reverse()

        return n_crop, n_disease, n_medicine


def main():
    crop_dict, disease_dict, medicine_dict, word_max_len = load_dict()
    ner_rule = NerBaseRule(crop_dict, disease_dict, medicine_dict, word_max_len)

    test = pd.read_csv('data/test/test.csv')
    submission = []
    for _, row in test.iterrows():
        n_crop, n_disease, n_medicine = ner_rule.ner(row['text'])
        submission.append([row['id'], str(n_crop), str(n_disease), str(n_medicine)])

    submission = pd.DataFrame(submission, columns=['id', 'n_crop', 'n_disease', 'n_medicine'])
    # submission.to_csv('submit/submit_forward_maximum_matching.csv', index=False)
    submission.to_csv('submit/submit_reverse_maximum_matching.csv', index=False)


if __name__ == '__main__':
    main()
