class State:
    def __init__(self):
        self.state = {
            k: v
            for k, v in self.__class__.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def __getattr__(self, name):
        if name in self.state:
            return self.state[name]
        elif name in dir(self):
            return getattr(self, name)
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == "state" or name in dir(self):
            super().__setattr__(name, value)
        else:
            self.state[name] = value


class Store(State):
    count = 4
    name = "jared"

    def increment(self):
        self.count += 1

    def decrement(self):
        self.count -= 1

    def hello(self):
        return f"hello {self.name}"


store = Store()

store.increment()
print(store.count)  # Prints: 5

store.decrement()
print(store.count)  # Prints: 4

print(store.hello())  # Prints: hello jared
