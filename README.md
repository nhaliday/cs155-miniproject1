# cs155-miniproject1

requirements.txt might be garbage. Haven't tested it. Will try to get a virtualenv
working.

Run `python3 src/train.py` to generate, score, and serialize a model (right now
  just random forest, will eventually split the actual modeling code into separate files
  with multiple models).

Run `python3 src/train.py ID1 ID2 ...` to load models ID1, ID2, etc. (you only
  have to give a prefix of the id) and run predict on the test data.
