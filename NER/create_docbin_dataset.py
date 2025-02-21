import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

# Sample data
texts = ["hello i'm omar my father name is hamed ","Hello, world!", "This is a test.", "Natural Language Processing with spaCy."]
docs = [nlp(text) for text in texts]

# Create a DocBin object
doc_bin = DocBin(docs=docs)

# Save to a file
doc_bin.to_disk("dataset.spacy")