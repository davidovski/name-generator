import training
import os
import sys

if __name__ == "__main__":

    # List all the directories containing country datasets to populate the countries dictionary
    countries = training.get_countries()

    for c, country in countries.items():
        print(f"processing {c}...", end="")
        sys.stdout.flush()

        letters = {}

        # get all the names in a country's dataset
        all_names = country.get_all()

        # iterate through all letters in the all of the names
        for name in all_names:

            # preprocess the name
            name = country.preprocess(name)

            for letter in name:
                if letter in letters:
                    letters[letter] += 1
                else:
                    letters[letter] = 1

        print(f" found {len(letters)} in {len(all_names)} names... ", end="")
        sys.stdout.flush()

        # sort the letters by occurrence
        letters_sorted = [l for l in letters]
        letters_sorted.sort()
        # output sorted letters to a file
        with open(os.path.join(country.path, "alphabet.txt"), "w") as file:
            for letter in letters_sorted:
                file.write(letter)
                file.write("\n")

        print("saved!")
        sys.stdout.flush()
