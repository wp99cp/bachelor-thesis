class singleton(object):
    instances = {}

    def __new__(cls, clz=None):
        if clz is None:
            # print ("Creating object for", cls)
            if cls.__name__ not in singleton.instances:
                singleton.instances[cls.__name__] = \
                    object.__new__(cls)
            return singleton.instances[cls.__name__]

        # print (cls.__name__, "creating", clz.__name__)
        singleton.instances[clz.__name__] = clz()
        singleton.first = clz
        return type(clz.__name__, (singleton,), dict(clz.__dict__))
