import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def start(self):
        self.start_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0.0
        else:
            return time.time() - self.start_time

    def reset(self):
        self.start_time = None


class TimerController:
    def __init__(self):
        self.timers = {}

    def start_timer(self, name):
        if name not in self.timers:
            self.timers[name] = Timer()
        self.timers[name].start()

    def reset_timer(self, name):
        if name in self.timers:
            self.timers[name].reset()

    def is_timer_running(self, name):
        if name in self.timers:
            return self.timers[name].start_time is not None
        else:
            return False

    def get_elapsed_time(self, name):
        if name in self.timers:
            return self.timers[name].get_elapsed_time()
        else:
            return None
