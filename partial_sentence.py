class PartialSentence:
    def __init__(self, in_word, last_hidden_state, out, predictions, score, attention_weights):
        self.in_word = in_word
        self.last_hidden_state = last_hidden_state
        self.out = out
        self.predictions = predictions
        self.score = score
        self.attention_weights = attention_weights
