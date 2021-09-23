import math

d_model = 512


def positional_encoding(pos, pe):
    for i in range(0, 512, 2):
        pe[0][i] = math.sin(pos / (1000 ** (2 * i / d_model)))
        pe[0][i + 1] = math.cos(pos / (1000 ** (2 * i / d_model)))

    return pe


def encode_word(pos, pe, y, pc):
    """
        This function will add the positional encoding to the
        embedding vector
        :param pos: position in the sentence
        :param pe: positional encoding vector
        :param y: word embedding
        :param pc: final embedding vector
        :return:
    """

    for i in range(0, 512, 2):
        pe[0][i] = math.sin(pos / (1000 ** (2 * i / d_model)))
        pc[0][i] = (y[0][i] * math.sqrt(d_model)) + pe[0][i]

        pe[0][i + 1] = math.cos(pos / (1000 ** (2 * i / d_model)))
        pc[0][i + 1] = (y[0][i + 1] * math.sqrt(d_model)) + pe[0][i + 1]
    return pc
print("ye")