
def make_trainable(net, val):
    for l in net.layers:
        l.trainable = val

