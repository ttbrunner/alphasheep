class Event:
    """
    Hamfisted event handler, so that the controller can send out update events to the GUI.

    NOTE: Using lambdas might have the potential for memory leaks - in a typical control flow you don't keep references to lambdas,
    so you can't really unsubscribe it later.
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


