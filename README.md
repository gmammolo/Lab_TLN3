
## Install

```python
source ./venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install uv
uv sync
```



### Bugfix

#### fix nltk/corpus/reader/framenet.py


`.venv/lib/python3.14/site-packages/nltk/corpus/reader/framenet.py`
```python
    def __getitem__(self, name):
        try:
            v = super().__getitem__(name)
            if isinstance(v, Future):
                return v._data()
            return v
        except Exception:
            return None
``` 