class Event:
    """
    Hamfisted event handler, so that the controller can send out update events to the GUI without depending on it.
    As usual - be careful when using lambdas.
    """

    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber_fn):
        self.subscribers.append(subscriber_fn)

    def unsubscribe(self, subscriber_fn):
        self.subscribers.remove(subscriber_fn)

    def notify(self):
        for subscriber_fn in self.subscribers:
            subscriber_fn()


