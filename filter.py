import math, random
random.seed(0)


train = [
    ("win a free prize now", 1),
    ("click here to claim your free money", 1),
    ("you won a free holiday click now", 1),
    ("cheap loans apply now", 1),
    ("claim your prize money today", 1),
    ("free money for you click here", 1),
    ("win cash now limited offer", 1),
    ("exclusive offer just for you click", 1),
    ("congratulations you won cash", 1),
    ("get rich fast free offer", 1),
    ("are we meeting for lunch today", 0),
    ("can you send me the lecture notes", 0),
    ("see you at the library tomorrow", 0),
    ("thanks for the help with the exam", 0),
    ("lets meet after the lecture", 0),
    ("my train is late see you soon", 0),
    ("did you finish the homework", 0),
    ("call me when you are home", 0),
    ("the meeting is moved to monday", 0),
    ("happy birthday have a great day", 0),
    ("wishes from your mother", 0),
    ("your account has a bonus waiting claim now", 1),
    ("act now and win a new phone", 1),
    ("lowest price guaranteed buy today", 1),
    ("you have been selected for a cash reward", 1),
    ("double your money in one week", 1),
    ("urgent your prize expires today click", 1),
    ("free gift card for the first winners", 1),
    ("earn money from home no experience", 1),
    ("special discount only today click here", 1),
    ("claim your reward before midnight", 1),
    ("love from your mother see you sunday", 0),
    ("your train leaves at five", 0),
    ("can you call your sister later", 0),
    ("wishes from all of us happy birthday", 0),
    ("dinner at our place on friday", 0),
    ("your notes from the lecture helped me", 0),
    ("i will be home late tonight", 0),
    ("mom says hello and sends her love", 0),
    ("the exam starts at nine tomorrow", 0),
    ("thanks for your message see you soon", 0),
]
test = [
    ("free cash prize click now", 1),
    ("you won money claim today", 1),
    ("limited offer get cash fast", 1),
    ("can we meet at the library", 0),
    ("send me the homework please", 0),
    
]

words = set()
for text, label in train:
    for word in text.split():
        words.add(word)
vocab = sorted(words)

index = {}

for i, word in enumerate(vocab):
    index[word] = i

def to_vector(text):
    v = [0.0] * len(vocab)
    for word in text.split():
        if word in index:
            v[index[word]] = 1.0
    return v


xs = [to_vector(t) for t, l in train]
ys = [l for t, l in train]

w = [0.0] * len(vocab)
b = 0.0
lr = 0.1

def sigmoid(z):
    return 1/(1+ math.exp(-z))

def predict(v):
    return sigmoid(sum(w[i] * v[i] for i in range(len(v)))+ b)

for epoch in range(1,2001):
    total_loss = 0.0
    for v,y in zip(xs,ys):
        p = predict(v) # forward
        d = p - y 
        for i in range(len(vocab)):
            w[i] = w[i] - lr * d * v[i]
        b = b - lr * d

#test
for text,label in test:
    p = predict(to_vector(text))
    guess = 1 if p > 0.5 else 0
    print(text)
    print('SPAM' if guess else 'ok' )
