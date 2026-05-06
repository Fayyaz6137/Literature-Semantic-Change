def verify_libraries():

    import gensim;
    print("gensim:", gensim.__version__)
    import numpy;
    print("numpy:", numpy.__version__)
    import sklearn;
    print("sklearn:", sklearn.__version__)
    import matplotlib;
    print("matplotlib:", matplotlib.__version__)
    import cade;
    print("cade: OK")
    import gutenbergpy;
    print("gutenbergpy: OK")
    print("All libraries OK ✓")
