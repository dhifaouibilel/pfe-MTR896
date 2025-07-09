
# import nltk
# nltk.download('all')
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

model = Doc2Vec.load(f"doc2vec/0_commit_message")
text = "This is a sample commit message"
embedding = model.infer_vector(word_tokenize(text.lower()))

print(embedding)