#!/usr/bin/env python3
"""Function that answers questions from a reference text"""

import cmd
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

bye = ['exit', 'quit', 'goodbye', 'bye']


def question_answer(question, reference):
    text = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(text)
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    question_tokens = tokenizer.tokenize(question)
    ref_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + ref_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) +\
        [1] * (len(ref_tokens) + 1)
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer is None or answer == '':
        answer = None
    return (answer)


def answer_loop(reference):
    """Function that answers questions from a reference text"""
    reference = reference

    class QABotCommand(cmd.Cmd):
        """QAbotCommand class"""
        prompt = "Q: "

        def precmd(self, line):
            """This method is called after the line has been input but before
                it has been interpreted. If you want to modify the input line
                before execution (for example, variable substitution)
                do it here."""
            if line.lower() in bye:
                print("A: Goodbye")
                self.do_bye(line)
                return ("bye")
            else:
                answer = question_answer(line, reference)
                if answer is None:
                    answer = "Sorry, I do not understand your question."
                print("A:", answer)
                return " "

        def do_bye(self, arg):
            """Method to exit and say goodbye"""
            return True

        def emptyline(self):
            """Called when an empty line is entered in response to the prompt.
            If this method is not overridden, it repeats the last nonempty
            command entered.
            """
            pass

    QABotCommand().cmdloop()
