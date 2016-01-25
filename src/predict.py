import sys, warnings
from pathlib import Path
from configparser import ConfigParser

import numpy as np
import pandas as pd

from sklearn.externals import joblib

config = ConfigParser()
config.read('settings.cfg')
paths = config['paths']

db = pd.read_csv(paths['modelDB'])
X = pd.read_csv(paths['test'], sep='|')

if len(sys.argv) == 1:
    sys.argv.append(db.loc[db.index[-1], 'IdMd5'])
for id in sys.argv[1:]:
    match = db.IdMd5.str.startswith(id)
    n = sum(match)
    if not n:
        warnings.warn("ID {} does not match any model.".format(id))
    elif n > 1:
        warnings.warn("Multiple models matching ID {}.".format(id))
    else:
        i = list(match).index(True)
        model_path = Path(paths['model']) / db.loc[i, 'SerializedModel']
        clf = joblib.load(str(model_path))
        y = pd.DataFrame(clf.predict(X), index=pd.Index(np.arange(1, len(X)+1), name='Id'), columns=['Prediction'])
        fname = str(Path(paths['predict']) / ('predict-' + model_path.stem + '.csv'))
        print(fname)
        y.to_csv(fname)
