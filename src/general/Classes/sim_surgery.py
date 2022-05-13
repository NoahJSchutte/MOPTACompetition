from src.general.Functions.general_functions import get_mean, lognorm_conditional_expectation_given_min

class SimSurgery:
    def __init__(
            self,
            scheduled_start_time: int,
            scheduled_day: str = None,
            is_emergency: bool = False
    ):
        self.scheduled_start_time = scheduled_start_time  # exclusive
        self.realized_start_time: int
        self.realized_duration: int
        self.finish_time: int       # inclusive. Note that if this is set it doesn't mean it has finished already
        self.has_started = False
        self.has_finished = False
        self.is_emergency = is_emergency
        self.is_cancelled = False

    def get_scheduled_start_time(self):
        return self.scheduled_start_time

    def set_realized_duration(self, realized_duration: int):
        self.realized_duration = realized_duration

    def get_realized_duration(self):
        return self.realized_duration

    def set_realized_start_time(self, realized_start_time: int):
        self.has_started = True
        self.realized_start_time = realized_start_time

    def get_realized_start_time(self):
        if self.has_started:
            return self.realized_start_time

    def set_finish_time(self):
        self.finish_time = self.realized_start_time + self.realized_duration

    def set_has_finished(self):
        self.has_finished = True

    def cancel_surgery(self):
        self.is_cancelled = True

    def get_finish_time(self):
        return self.finish_time

    def get_realized_waiting_time(self):
        return self.realized_start_time - self.scheduled_start_time

    def get_expected_finish_time(self, distribution, parameters, current_time):
        if distribution == 'exponential':
            return get_mean(distribution, parameters)
        elif distribution == 'lognormal':
            if abs(self.realized_start_time - current_time) < 0.0001:
                return get_mean(distribution, parameters)
            elif current_time < self.realized_start_time:
                print("error")
                raise Exception("Shouldn't happen")
            else:
                return lognorm_conditional_expectation_given_min(*parameters, current_time-self.realized_start_time)


