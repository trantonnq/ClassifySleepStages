class SleepWakePerformance(object):
    def __init__(self, accuracy, wake_correct, sleep_correct, kappa, auc, sleep_predictive_value,
                 wake_predictive_value):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.sleep_correct = sleep_correct
        self.kappa = kappa
        self.auc = auc
        self.wake_predictive_value = wake_predictive_value
        self.sleep_predictive_value = sleep_predictive_value


class ThreeClassPerformance(object):
    def __init__(self, accuracy, wake_correct, rem_correct, nrem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.rem_correct = rem_correct
        self.nrem_correct = nrem_correct
        self.kappa = kappa

class FourClassPerformance(object):
    def __init__(self, accuracy, wake_correct, n1_n2_correct, n3_correct, rem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.n1_correct = n1_n2_correct
        self.n3_correct = n3_correct
        self.rem_correct = rem_correct
        self.kappa = kappa


class FiveClassPerformance(object):
    def __init__(self, accuracy, wake_correct, n1_correct, n2_correct, n3_correct, rem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.n1_correct = n1_correct
        self.n2_correct = n2_correct
        self.n3_correct = n3_correct
        self.rem_correct = rem_correct
        self.kappa = kappa

