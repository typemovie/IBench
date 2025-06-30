class DataInstance(object):
    def __init__(self, file):
        self._datas = file.get("datas", None)
        self._tags = file.get("tags", None)
        self._category = self._tags.get("category", None)

    def imagenums(self):
        pass

    @property
    def datas(self):
        return self._datas

    @property
    def tags(self):
        return self._tags

    @property
    def category(self):
        return self._category


