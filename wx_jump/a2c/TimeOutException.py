
class TimeOutException(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self) -> str:
        return self.message