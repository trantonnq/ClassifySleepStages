from enum import Enum


class SleepWakeLabel(Enum):
    wake = 0
    sleep = 1


class ThreeClassLabel(Enum):
    wake = 0
    nrem = 1
    rem = 2

class FourClassLabel(Enum):
    wake = 0
    n1_n2 = 1
    n3_n4 = 2
    rem = 3

class FiveClassLabel(Enum):
    wake = 0
    n1 = 1
    n2 = 2
    n3_n4 = 3
    rem = 4

class SixClassLabel(Enum):
    wake = 0
    n1 = 1
    n2 = 2
    n3 = 3
    n4 = 4
    rem = 5