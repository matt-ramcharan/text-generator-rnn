from __future__ import print_function
import numpy as np


# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
	# starting with random character
	ix = [np.random.randint(vocab_size)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocab_size))
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		#tqdm.write(ix_to_char[ix[-1]], end="")
		#tqdm.write(ix_to_char[ix[-1]], file=open("Pott.txt","a"), end="")
		pred = model.predict(X[:, :i + 1, :])[0]
		ix = np.argmax(pred, 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

# method for preparing the training data
def load_data(data_dir, seq_length):
	file = open(data_dir, 'r', encoding="utf-8")
	data=file.read()
	#if len(data) % seq_length != 0:
	#	raise ValueError('Data is not divisible by Sequence Length (Make divisible by ' + str(len(data)) + ')')

	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	X = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE), dtype=np.bool)
	y = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE), dtype=np.bool)
	for i in range(0, int(len(data)/seq_length)):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
		X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
		y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char