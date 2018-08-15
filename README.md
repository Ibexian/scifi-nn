# scifi-nn
Explorations in LSTM with a Sci-fi data set

For more info see [notes](notes.md)

## Commands
#### To begin training:

```bash
python localKeras.py train
```

#### To continue training:
```bash
python localKeras.py continue
```

_searches for the highest numbered model in the main directory - e.g. model-13.hdf5_

#### To run a visual test of the model's accuracy:

```bash
python localKeras.py test
```

_as with `continue` this searches for the highest numbered model_

#### To generate text based on your model:

```bash
python generateSciFi.py --length 1000 --model 13 --seed "It all began with"
```

(`length` defaults to `140` and `model` defaults to `'04'`)