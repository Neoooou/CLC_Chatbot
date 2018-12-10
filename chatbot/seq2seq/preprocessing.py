# author Ran
import pandas as pd
import numpy as np
import tensorflow as tf
import re, time, os
import random
random.seed(1990)





class preprocessing:
    def __init__(self):
        self.get_questions_answers()
        self.clean_data()
        self.short_questions, self.short_answers = self.filter_data()
        # compute vocabluary from data
        self.questions_vocab_to_int, self.answers_vocab_to_int = \
            self.prepare_vocabs(self.short_questions, self.short_answers)
        self.questions_int_to_vocab = {v_i: v for v, v_i in self.questions_vocab_to_int.items()}
        self.answers_int_to_vocab = {v_i: v for v, v_i in self.answers_vocab_to_int.items()}
        self.questions_int, self.answers_int = self.encode_data(self.short_questions,
                                                                self.short_answers,
                                                                self.questions_vocab_to_int,
                                                                self.answers_vocab_to_int)
        # pad sentence to fixed length
        self.encoder_input_data = np.array(
            self.pad_sentence(
                self.questions_int,
                self.questions_vocab_to_int,
                mode="pre")
        )
        self.decoder_input_data = np.array(
            self.pad_sentence(
                self.answers_int,
                self.answers_vocab_to_int)
        )
        self.num_encoder_tokens = len(self.questions_vocab_to_int)
        self.num_decoder_tokens = len(self.answers_vocab_to_int)
        num_lines = self.decoder_input_data.shape[0]
        self.max_length = self.decoder_input_data.shape[1]
        self.decoder_target_data = np.zeros(
            (num_lines, self.max_length, self.num_decoder_tokens),
            dtype="float32")
        for i, target_text in enumerate(self.decoder_input_data):
            # decoder_target_data is one time step ahead of decoder_input_data
            for t, word in enumerate(target_text):
                if t > 0:
                    self.decoder_target_data[i, t - 1, word] = 1

    def clean_text(self, text):
        # clean text by removing unnecessary characters and altering the format of words
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'d", "  would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        return text


    # get all the question and answer lists from the corpus
    def get_questions_answers(self):
        lines = open("chatbot/corpus/movie_lines.txt", encoding= "utf-8", errors="ignore").read().split("\n")
        conv_lines = open("chatbot/corpus/movie_conversations.txt",encoding="utf-8", errors="ignore").read().split("\n")
        id2line = {}
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
        convs = []
        for line in conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            convs.append(_line.split(","))

        #sort sentences into questions and answers
        questions = []
        answers = []
        for conv in convs:
            for i in range(len(conv)-1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i+1]])
        self.questions = questions
        self.answers = answers
        # compare lengths of questions and answers
        # print("length of total questions: ", len(questions))
        # print("length of total answers: ", len(answers))


    def clean_data(self):
        self.questions = [self.clean_text(question) for question in self.questions]
        self.answers = [self.clean_text(answer) for answer in self.answers]

    def filter_data(self):
        # Find the length of sentences
        lengths = []
        for question in self.questions:
            lengths.append(len(question.split()))
        for answer in self.answers:
            lengths.append(len(answer.split()))
        # Create a dataframe so that the values can be inspected
        lengths = pd.DataFrame(lengths, columns=['counts'])
        lengths.describe()
        # print(np.percentile(lengths, 80))
        # print(np.percentile(lengths, 85))
        # print(np.percentile(lengths, 90))
        # print(np.percentile(lengths, 95))
        # print(np.percentile(lengths, 99))

        # Remove questions and answers that are longer than 20 words or shorter than 2 words
        min_line_length = 2
        max_line_length = 20
        # filter out the questions that are too long / short
        i = 0
        short_questions_temp = []
        short_answers_temp = []
        for question in self.questions:
            if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
                short_questions_temp.append(question)
                short_answers_temp.append(self.answers[i])
            i += 1
        # filter out the answers that are too long / short
        i = 0
        short_questions = []
        short_answers = []
        for answer in short_answers_temp:
            if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
                short_answers.append(answer)
                short_questions.append(short_questions_temp[i])
            i += 1
        qas = zip(short_questions, short_answers)
        # Randomly choose 5000 lines of data from questions and answers respectively
        # for the purpose of avoiding memory error
        random_samples = random.sample(list(qas), 5000)
        short_questions,  short_answers = [], []
        for i in range(len(random_samples)):
            short_questions.append(random_samples[i][0])
            short_answers.append(random_samples[i][1])
        # for i in range(100):
        #     print("q:", short_questions[i])
        #     print("a:", short_answers[i])
        #     print("---")
        # print("length of short questions:", len(short_questions))
        # print("length of short answers:", len(short_answers))
        return short_questions, short_answers

    def prepare_vocabs(self,short_questions, short_answers):
        # compare the number of lines we will use with the total number of lines
        # print("# of questions: ", len(short_questions))
        # print("# of answers: ", len(short_answers))
        # print("% data of used :{}".format(round((len(short_answers) / len(self.answers)), 4) * 100))
        # create a dictionary for frequency of vocabulary
        vocab = {}
        for question in short_questions:
            for word in question.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        for answer in short_answers:
            for word in answer.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        # remove rare words from the vocabulary, we aim to replace fewer than 5% of words with <UNK>
        threshold = 0
        count = 0
        for k, v in vocab.items():
            if v >= threshold:
                count += 1
        # print("size of total vocab:", len(vocab))
        # print("size of vocab we will use", count)
        # In this case, we want ot use a different sizes for the source and the target
        # we can set different threshold values
        # we wiil create dictionaries to provide a unique integer for each word
        questions_vocab_to_int = {}
        word_num = 0
        for word, count in vocab.items():
            if count >= threshold:
                questions_vocab_to_int[word] = word_num
                word_num += 1
        answers_vocab_to_int = {}
        word_num = 0
        for word, count in vocab.items():
            if count >= threshold:
                answers_vocab_to_int[word] = word_num
                word_num += 1
        codes = ["<PAD>", "<EOS>", "<UNK>", "<GO>"]
        for code in codes:
            questions_vocab_to_int[code] = len(questions_vocab_to_int)
            answers_vocab_to_int[code] = len(answers_vocab_to_int)

        return questions_vocab_to_int, answers_vocab_to_int

    def encode_data(self, short_questions,
                    short_answers,
                    questions_vocab_to_int,
                    answers_vocab_to_int):
        # Add the end of sentence("<EOS>") to the end of every answer
        # ADD the start of sentence("<GO>") to the start of every answer
        for i in range(len(short_answers)):
            short_answers[i] = "<GO> " + short_answers[i] + " <EOS>"
        # Convert the text into integers, if the word is not in the respective vocabulary, replace with <UNK>
        questions_int = []
        for question in short_questions:
            ints = []
            for word in question.split():
                if word not in questions_vocab_to_int:
                    ints.append(questions_vocab_to_int["<UNK>"])
                else:
                    ints.append(questions_vocab_to_int[word])
            questions_int.append(ints)
        answers_int = []
        for answer in short_answers:
            ints = []
            for word in answer.split():
                if word not in answers_vocab_to_int:
                    ints.append(answers_vocab_to_int["<UNK>"])
                else:
                    ints.append(answers_vocab_to_int[word])
            answers_int.append(ints)
        # Calculate what percentage of all words have been replaced with <UNK>
        word_count = 0
        unk_count = 0
        for question in questions_int:
            for val in question:
                if val == questions_vocab_to_int["<UNK>"]:
                    unk_count += 1
                word_count += 1
        for answer in answers_int:
            for val in answer:
                if val == answers_vocab_to_int["<UNK>"]:
                    unk_count += 1
                word_count += 1
        unk_ratio = round(unk_count / word_count, 4) * 100
        # print("Total number of words:", word_count)
        # print("Total number of unknown words(<UNK>):", unk_count)
        # print("Percentage of words that are <UNK>: {}%".format(round(unk_ratio, 3)))
        return questions_int, answers_int
    def pad_sentence(self,sentences,
                     vocab_to_int,
                     max_length = 22,
                     mode="post"):
        for i in range(len(sentences)):
            pad_times = max_length - len(sentences[i])
            if mode == "post":
                sentences[i].extend(np.repeat(vocab_to_int["<PAD>"], pad_times))
            elif mode == "pre":
                new_sent = list(np.repeat(vocab_to_int["<PAD>"], pad_times))
                new_sent.extend(sentences[i])
                sentences[i] = new_sent
        return sentences




if __name__ == "__main__":
    pp = preprocessing()













