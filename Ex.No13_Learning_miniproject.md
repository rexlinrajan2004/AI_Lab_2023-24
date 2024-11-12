# Ex.No: 10  NEXT WORD PREDICTION USING LSTM
### DATE: 24.10.2024                                                                           
### REGISTER NUMBER : 212222220034
### AIM: 
To build an LSTM-based model for accurate next-word prediction, enhancing applications like auto-correction and search suggestions.
###  Algorithm:

1. Preprocess the text data by tokenizing sentences, creating sequences, and encoding words into integers.
2. Create input sequences from text data and pad them to a consistent length.
3. Define an LSTM model with an embedding layer, LSTM layer, dense layer, and output layer with softmax activation.
4. Compile the model using categorical cross-entropy as the loss function and an optimizer like Adam.
5. Train the model on the input sequences and their corresponding labels.
6. For prediction, input a partial sentence, use the model to predict the next word, and decode the prediction to get the next word.

### Program:
```
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
len(tokenizer.word_index)
input_sequences = []
for sentence in faqs.split('\n'):
tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
for i in range(1,len(tokenized_sentence)):
input_sequences.append(tokenized_sentence[:i+1])
input_sequences
max_len = max([len(x) for x in input_sequences])
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
padded_input_sequences
X = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]
len(y)
X.shape
y.shape
from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=283)
y.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
model = Sequential()
model.add(Embedding(282, 100, input_length=56))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(283, activation='softmax')) # Change this line
# Recompile the model with the updated architecture
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(X,y,epochs=100)
import time
text = "add your text"
for i in range(5):
token_text = tokenizer.texts_to_sequences([text])[0]
# Assuming max_len should be 56 to match the model's input_length
padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
pos = np.argmax(model.predict(padded_token_text))
for word,index in tokenizer.word_index.items():
if index == pos:
text = text + ' ' + word
print(text)
time.sleep(2)
def calculate_perplexity(model, X, y):
cross_entropy = model.evaluate(X, y, verbose=0)
perplexity = np.exp(cross_entropy)
return perplexity
# Calculate accuracy and perplexity on the test set
accuracy = model.evaluate(X, y, verbose=0)[1]
print(f'Accuracy: {accuracy * 100:.2f}%')
def calculate_perplexity(model, X, y):
cross_entropy = model.evaluate(X, y, verbose=0)
perplexity = np.exp(cross_entropy)
return perplexity
# Calculate accuracy and perplexity on the test set
perplexity = calculate_perplexity(model, X, y)
# Assuming you want to print the first perplexity value
print(f'Perplexity: {perplexity[0]:.2f}')
```
### Output:

![ot1](https://github.com/user-attachments/assets/dc4dcc1e-63eb-4335-a411-47a78921456f)
![image](https://github.com/user-attachments/assets/9061c5bd-47d7-4368-93b6-a1a4c0a2b525)
![image](https://github.com/user-attachments/assets/6a2c8f98-5d32-4c2c-b4ef-50bc21878c56)


### Result:
Thus the system was trained successfully and the prediction was carried out.
