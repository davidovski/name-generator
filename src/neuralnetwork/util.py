import sqlite3
import tensorflow as tf
from pincelate import Pincelate
from multiprocessing import Value
import random

"""Atomic Number
A multiprocessing-safe Number that you can increment and set a value to.
"""
class AtomicNumber:
    def __init__(self, initial=0):
        # set the initial value of the number
        self._v = Value("d", initial)

        # use python thread locks
        self._lock = self._v.get_lock()

    def set(self, num):
        with self._lock:
            # set the value of the number and return it
            self._v.value = num
            return self._v.value

    def increment(self, num=1):
        with self._lock:
            # increase the value of the number and then return it
            self._v.value += num
            return self._v.value

    def get(self):
        return self._v.value


""" Capitalise the first letter of a name
    Args:
        name: (str) the name to transform
    Returns:
        name: (str) the output name
"""


def capitalise(name):
    return str(name).capitalize()


""" uncapitalise the first letter of a name
    Args:
        name: (str) the name to transform
    Returns:
        name: (str) the output name
"""


def uncapitalise(name):
    return str(name).lower()


# import pincelate

# tell tensorflow to only log ERRORS rather than all problems
tf.get_logger().setLevel('ERROR')

pin = Pincelate()

""" Transliterate names into their phonetic spelling 
    Args:
        name: (str) the name to transliterate
    Returns:
        name: (str) the pronunciation of the name as an arpabet list
"""


def transliterate(name):
    output = []

    # iterate through each word and sound out separately
    for word in name.split(" "):
        try:
            for phoneme in pin.soundout(word):
                output.append(phoneme)
        except Exception as e:
            output.append(word)
        output.append(" ")

    # remove the trailing " "
    output.pop()

    return output


""" Transliterate phonetic spellings back into names
    Args:
        name: (str) the pronunciation of the name as an arpabet list
    Returns:
        name: (str) a guessed spelling of the given phoneme list
"""


def reverse_transliterate(name):
    words = []

    current_word = []
    # iterate through all phonemes, spliting groups between " " into words
    for phoneme in name:
        if phoneme == " ":
            words.append(current_word)

            # reset current word
            current_word = []
        else:
            current_word.append(phoneme)

    # add whats left of the current word to the list of words
    words.append(current_word)

    # spell each word separately
    spellings = [pin.spell(word) for word in words]
    return " ".join(spellings)


"""Load random line from a given filename
"""


def get_random_line(file_name):
    total_bytes = os.stat(file_name).st_size
    random_point = random.randint(0, total_bytes)
    file = open(file_name)
    file.seek(random_point)
    file.readline()  # skip this line to clear the partial line
    return file.readline()


""" Concatenate elemnts of a list
    Args:
        name: (str[]) the letters of a name in an array 
    Returns:
        name: (str) 
"""


def deserialise(name):
    return "".join(name)


""" return a random item from a dictionary of weighted items
    Args:
        weights: (dict) a dictionary containing the items as keys and float values as weights
    Returns:
        item: a random item selected from the dictionary
"""


def weighted_choice(weights):
    # choose a randm number between 0 and 1
    choice = random.uniform(0.0, 1.0)

    # iterate through weights, subtracting the weight from the choice each time
    for item, weight in weights.items():
        choice -= weight

        # if we are currently on the chosen item, return it
        if choice < 0:
            return item

    # in case the input dictionary is incorrectly setup, return the last item
    return list(weights)[-1]


def create_table(database_path):
    # connect to the database and get the cursor
    connection = sqlite3.connect(database_path)

    # always close the connection at the end
    with connection:
        cursor = connection.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS names (\
                                Name TEXT,\
                                Origin TEXT,\
                                Category TEXT\
                                )")

        # commit the changes, saving them
        connection.commit()
