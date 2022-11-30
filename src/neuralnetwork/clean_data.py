import training
import os
import sys

if __name__ == "__main__":

    countries = training.get_countries()

    for c, country in countries.items():
        if c == "uk" or c == "usa":

            # iterate through each dataset separately
            for dataset, path in country.datasets.items():

                # print information
                print(f"filtering through {c}'s {dataset}...")
                sys.stdout.flush()

                names = country.get_names(dataset)

                # store the names that are valid in a seperate list
                names_output = []

                # load the alphabet file for the country
                alphabet = []

                with open(os.path.join(country.path, "alphabet.txt"), "r") as file:
                    for l in file.read().split("\n"):
                        alphabet.append(l)

                c = 0
                t = len(names)
                # iterate through names in the dataset
                for name in names:
                    name = country.preprocess(name)

                    valid = True

                    # invalidate the name if a single letter is not in the alphabet
                    for letter in name:
                        if not letter in alphabet:
                            valid = False
                            break

                    if valid:
                        names_output.append(name)

                    c += 1
                    if c % 128 == 0:
                        print(f"\r{c}/{t}", end="")

                # print how many names are left
                print(f"kept {len(names_output)}/{len(names)} names")

                # save dataset
                with open(path, "w") as file:
                    for name in names_output:
                        file.write(name)
                        file.write("\n")
