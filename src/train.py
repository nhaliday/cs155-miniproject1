import sys, hashlib, tempfile, shutil
from pathlib import Path
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.externals import joblib

from sklearn import ensemble, cross_validation
from sklearn.grid_search import GridSearchCV

config = ConfigParser()
config.read('settings.cfg')
paths = config['paths']

data = pd.read_csv(paths['train'], sep='|')

seed = 3128
rng = np.random.RandomState(seed)
time = datetime.utcnow().isoformat()
metadata = dict(id=None, seed=seed, time=time, file=None, score=None, dev=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

clf = ensemble.RandomForestClassifier(n_estimators=500, random_state=rng)
scores = cross_validation.cross_val_score(
    clf, X, y,
    cv=cross_validation.KFold(len(data), n_folds=5, shuffle=True, random_state=rng),
)
metadata['score'] = scores.mean()
metadata['dev'] = scores.std()

clf.fit(X, y)

tmp = tempfile.NamedTemporaryFile(delete=False)
joblib.dump(clf, tmp.name, compress=7)

m = hashlib.md5()
m.update(metadata['seed'].to_bytes(10, sys.byteorder))
m.update(bytes(metadata['time'], 'UTF-8'))
with open(tmp.name, 'rb') as model:
    m.update(model.read())

metadata['id'] = m.hexdigest()
metadata['file'] = 'RandomForestClassifier-{}-{}.pkl'.format(metadata['time'], metadata['id'])
tmp.close()
shutil.move(tmp.name, str(Path(paths['model']) / metadata['file']))

with open(paths['modelDB'], 'a') as db:
    print("{id},{seed},{time},{file},{score},{dev}".format(**metadata), file=db)
