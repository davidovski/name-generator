import os
import types
import json
import random

from util import *
from rnn import *

cuda = False
num_processes = 12


class Country:
    def __init__(self, path):
        self.path = path
        self.datasets = {
            "female": os.path.join(path, "female.txt"),
            "male": os.path.join(path, "male.txt"),
            "surname": os.path.join(path, "surname.txt"),
        }

        # initialise the pre and post proccess function lists
        self.pre_process = []
        self.post_process = []

        # load the data file
        self.load_data()

        # load the alphabet file
        self.alphabet = self.load_alphabet()

        # initialise the rnn models
        hidden_size = 128
        self.rnn = {}

        for dataset in self.datasets:
            self.rnn[dataset] = RNN(
                len(self.alphabet), hidden_size, len(self.alphabet))

    """ Load the alphabet from the alphabet file
        Returns:
            alphabet: (str[]) the list of the letters/characters to use while training
    """

    def load_alphabet(self):
        alphabet_path = os.path.join(self.path, "alphabet.txt")

        # check if the alphabet file exists, if not, raise an exception
        if os.path.exists(alphabet_path):
            with open(alphabet_path, "r") as alphabet_file:
                # Split the file by lines: on letter/character should be on each line
                letters = alphabet_file.read().split("\n")
                return letters
        else:
            raise Exception(
                f"The alphabet file {alphabet_path} could not be found")
            return []

    """ load the data from the data file
    """

    def load_data(self):
        data_path = os.path.join(self.path, "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as data_file:
                j = json.loads(data_file.read())

                # match the imported global function with the ones listed in the json file
                for pre in j["pre"]:
                    if pre in globals():
                        func = globals()[pre]

                        # check if the requested object is a function
                        if type(func) is types.FunctionType:
                            self.pre_process.append(func)
                        else:
                            raise Exception(
                                f"The function '{pre}' is not a function")
                    else:
                        # If the function was not loaded, throw an exception
                        raise Exception(
                            f"The function '{pre}' was not loaded or does not exist")

                for post in j["post"]:
                    if post in globals():
                        func = globals()[post]

                        # check if the requested object is a function
                        if type(func) is types.FunctionType:
                            self.post_process.append(func)
                        else:
                            raise Exception(
                                f"The function '{post}' is not a function")
                    else:
                        # If the function was not loaded, throw an exception
                        raise Exception(
                            f"The function '{post}' was not loaded or does not exist")

        else:
            # load the default pre and post processing functions
            self.pre_process = [uncapitalise]
            self.post_process = [deserialise, capitalise]

    """ List all the names from a given category file
        Args:
            category: (str) the category to select names from
        Returns:
            data: (str[]) an array containing all of the names from the given category file
    """

    def get_names(self, category):
        with open(self.datasets[category], "r") as datafile:
            return [name for name in datafile.read().split("\n")]

    """ List all names in all categories
        Returns:
            data: (str[]) an array with all of the names in this country's datasets
    """

    def get_all(self):
        return [name for k in self.datasets for name in self.get_names(k)]

    """ Pre-process a name for training
        Args:
            name: the name loaded from the dataset
        Returns:
            name: the name after being processed
    """

    def postprocess(self, name):
        for f in self.post_process:
            name = f(name)
        return name

    """ Post-process a name after sampling
        Args:
            name: the name output from the recurrent neural network
        Returns:
            name: the name after being processed
    """

    def preprocess(self, name):
        for f in self.pre_process:
            name = f(name)
        return name

    """ Train a neural network on the given dataset
        Args:
            category: (str) the category to sample training names from
    """

    def train(self, category):
        # select the RNN model to be training on
        rnn = self.rnn[category]

        # load names from that dataset and pre proccess them
        print("preprocessing names...")
        names = [self.preprocess(name) for name in self.get_names(category)]
        print(f"processed {len(names)} names!")

        # calculate optimum number of iterations (using 80% of whole dataset)
        iters = int(len(names) * 0.8)

        # start training
        learn_names(rnn, names, self.alphabet, iterations=iters,
                    num_processes=num_processes)

    """ Sample a name from the neural network with a given starting letter
        Args:
            category: (str) the category to sample generated names from
        Returns:
            name: the output from the neural network
    """

    def sample(self, category, start_letter):

        # select the RNN model to be sampling from
        rnn = self.rnn[category]

        # set the random factor of the RNN to randomise names that are generated
        rnn.random_factor = 0.7

        # call the rnn sample function to generate a single name
        name = sample(rnn, self.alphabet, start_letter)

        # post process the name and return
        return self.postprocess(name)

    """ Load the rnn from its file
        Args:
            category: (str) the category to load
            parent_directory: (str) where to find the model
    """

    def load_rnn(self, category, parent_directory):
        model_file = os.path.join(parent_directory, f"{category}.pt")
        self.rnn[category] = torch.load(model_file)

    """ Save the rnn of a given category to its file
        Args:
            category: (str) the category to save
            parent_directory: (str) the directory to save the model file to  
    """

    def save_rnn(self, category, parent_directory):
        rnn = self.rnn[category]
        model_file = os.path.join(parent_directory, f"{category}.pt")
        torch.save(rnn, model_file)


def get_countries():
    return {
        country: Country(os.path.join(countries_path, country)) for country in os.listdir(countries_path) if os.path.isdir(os.path.join(countries_path, country))
    }


""" train all of the datasets from a specific country
    Args:
        country: (Country) 
"""


def train_country(country, name):
    datasets = country.datasets
    for dataset in datasets:
        print(f"Training {dataset} in {name}")
        country.train(dataset)

        print(f"Finished training on {dataset}... saving...", end="")
        path = os.path.join("data", "models", name)

        # check if the path already exists before trying to make directories
        if not os.path.exists(path):
            os.makedirs(path)

        country.save_rnn(dataset, path)
        print("saved!")


def sample_country(country, country_name, number_of_samples=10000):

    datasets = country.datasets
    for dataset in datasets:

        # ensure that the model exists before sampling
        path = os.path.join("data", "models", country_name)
        if os.path.exists(os.path.join(path, dataset + ".pt")):

            # load the country's rnn
            country.load_rnn(dataset, path)

            # load the names from the country's dataset, and pre-process them
            names = [country.preprocess(name)
                     for name in country.get_names(dataset)]

            # make a dictionary full of start letters and their frequency
            start_letters = {}

            for name in names:
                if len(name) > 0:
                    start_letter = name[0]

                    # if the start letter isn't already in the dictionary, add it with value 1
                    if start_letter in start_letters:
                        start_letters[start_letter] += 1
                    else:
                        start_letters[start_letter] = 1

            # turn each integer count into a float where: letter_weight=frequency/total_names
            total = len(names)

            for letter in start_letters:
                weight = float(start_letters[letter] / total)
                start_letters[letter] = weight

            # sample names from the RNN
            sampled_names = []

            for i in range(number_of_samples):
                try:
                    letter = weighted_choice(start_letters)
                    sample = country.sample(dataset, letter)
                    sampled_names.append(sample)
                except:
                    pass

            # remove duplicate names
            sampled_names = list(dict.fromkeys(sampled_names))

            # create a sqlite connection
            connection = sqlite3.connect(database)

            # always close the connection when finished
            with connection:
                cursor = connection.cursor()
                for name in sampled_names:
                    sql = "INSERT INTO names(Name, Origin, Category) VALUES(?, ?, ?)"

                    # insert the current name and options into the database
                    cursor.execute(sql, (name, country_name, dataset))

                # commit changes and save the database
                connection.commit()

            print(
                f"Saved {len(sampled_names)} names for {country_name}/{dataset}")

        else:
            print(f"the model: {country_name}/{dataset} was not found.")


countries_path = "data/datasets"
database = os.path.join("data", "names.db")
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # allow processes on this model to share memory
        torch.multiprocessing.set_start_method('spawn')

        # List all the directories containing country datasets to populate the countries dictionary
        countries = get_countries()

        country_count = len(countries)
        # Display debug information
        print(f"Loaded {country_count} countries!")

        # list all countries in neat collumns
        collumns = 4
        width = 14
        i = 0
        for country in countries:
            i += 1

            # print the country and then its index
            print(country, end="")

            # organise into rows and collumns
            if i % collumns == 0:
                print("")
            else:
                # separate collumns with spaces
                print(" " * (width - len(country)), end="")

        # keep asking until the country selection is valid
        good_selection = False
        while not good_selection:
            # prompt user to select a country to train, or train all
            country_selection = input(
                "select the name of a country to train on, or (all) to train on all countries: ")

            good_selection = True
            selected_countries = []

            # if the user selected all, then add all countries to list, if not, add the selected country
            if country_selection.lower() == "all":
                [selected_countries.append(country) for country in countries]
            elif country_selection.lower() in countries:
                selected_countries.append(country_selection)
            else:
                print("Country not found, try again")
                good_selection = False

            choice = input("(t)rain on data, or (s)ample from weights?")

            if choice.lower()[0] == "t":
                for country in selected_countries:
                    train_country(countries[country], country)

            elif choice.lower()[0] == "s":
                create_table(database)
                for country in selected_countries:
                    sample_country(countries[country], country)
