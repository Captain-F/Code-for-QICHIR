import pickle

def save_pickle(data, name):
  path = "data/to/path/"+name+'.pickle'
  with open(path, 'wb') as f:
    pickle.dump(data, f)

def read_pickle(name):
  path = "data/to/path/" + name + ".pickle"
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data