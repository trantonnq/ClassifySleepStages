import numpy as np

from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.sleep_label import SleepWakeLabel
from source.analysis.setup.sleep_label import ThreeClassLabel, FourClassLabel, FiveClassLabel


class SleepLabeler(object):

    @staticmethod
    def label_sleep_wake(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            if value > 0:
                converted_value = SleepWakeLabel.sleep.value
            else:
                converted_value = SleepWakeLabel.wake.value
            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def label_three_class(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            if value == 0:
                converted_value = ThreeClassLabel.wake.value
            elif value == 5:
                converted_value = ThreeClassLabel.rem.value
            else:
                converted_value = ThreeClassLabel.nrem.value

            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def label_four_class(raw_sleep_wake):
        labeled_sleep = []
        raw_file = open("raw_4.txt","a")     

        for value in raw_sleep_wake:
            if value == 0:
                converted_value = FourClassLabel.wake.value
            elif value == 1 :
                converted_value = FourClassLabel.n1_n2.value
            elif value == 2:
                converted_value = FourClassLabel.n1_n2.value
            elif value == 3:
                converted_value = FourClassLabel.n3_n4.value
            elif value == 4:
                converted_value = FourClassLabel.n3_n4.value
            elif value == 5:
                converted_value = FourClassLabel.rem.value
            else:
                converted_value = -99

            raw_file.write(str(value)+","+str(converted_value)+"\n")

            labeled_sleep.append(converted_value)

        raw_file.close()
        
        return np.array(labeled_sleep)

    @staticmethod
    def label_five_class(raw_sleep_wake):
        labeled_sleep = [] 
        
        for value in raw_sleep_wake:
            if value == 0:
                converted_value = FiveClassLabel.wake.value
            elif value == 1:
                converted_value = FiveClassLabel.n1.value
            elif value == 2:
                converted_value = FiveClassLabel.n2.value
            elif value == 3:
                converted_value = FiveClassLabel.n3_n4.value
            elif value == 4:
                converted_value = FiveClassLabel.n3_n4.value
            elif value == 5:
                converted_value = FiveClassLabel.rem.value
            else:
                converted_value = -99
           
            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def label_six_class(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            if value == 0:
                converted_value = SixClassLabel.wake.value
            elif value == 1:
                converted_value = SixClassLabel.n1.value
            elif value == 2:
                converted_value = SixClassLabel.n2.value
            elif value == 3:
                converted_value = SixClassLabel.n3.value
            elif value == 4:
                converted_value = SixClassLabel.n4.value
            else:
                converted_value = SixClassLabel.rem.value


            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)    

    @staticmethod
    def label_one_vs_rest(sleep_wake_labels, positive_class):
        labeled_sleep = []

        for value in sleep_wake_labels:
            if value == positive_class:
                converted_value = 1
            else:
                converted_value = 0

            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)



    @staticmethod
    def convert_three_class_to_two(raw_performance: RawPerformance):
        raw_performance.true_labels = SleepLabeler.label_sleep_wake(raw_performance.true_labels)
        number_of_samples = np.shape(raw_performance.class_probabilities)[0]
        for index in range(number_of_samples):
            raw_performance.class_probabilities[index, 1] = raw_performance.class_probabilities[index, 1] + \
                                                            raw_performance.class_probabilities[index, 2]
        raw_performance.class_probabilities = raw_performance.class_probabilities[:, :-1]

        return raw_performance
