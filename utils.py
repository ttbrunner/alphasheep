
class Event:
    # Hamfisted event handler, so that the controller can send out

    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber_fn):
        self.subscribers.append(subscriber_fn)

    def unsubscribe(self, subscriber_fn):
        # NOTE: Will probably not work with lambda, possible memory leak. Try not to use this with lambda, OK?
        self.subscribers.remove(subscriber_fn)

    def notify(self):
        for subscriber_fn in self.subscribers:
            subscriber_fn()

