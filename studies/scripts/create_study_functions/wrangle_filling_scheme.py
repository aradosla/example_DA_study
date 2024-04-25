import json

import numpy as np


def reformat_filling_scheme_from_lpc(filling_scheme_path):
    """
    Alternative to the function above, as sometimes the injection information is not present in the
    file. Not optimized but works.
    """

    # Load the filling scheme directly if json
    with open(filling_scheme_path, "r") as fid:
        data = json.load(fid)

    # Take the first fill number
    fill_number = list(data["fills"].keys())[0]

    # Do the conversion (Matteo's code)
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    l_lines = data["fills"][f"{fill_number}"]["csv"].split("\n")
    for idx, line in enumerate(l_lines):
        # First time one encounters a line with 'Slot' in it, start indexing
        if "Slot" in line:
            # B1 is initially empty
            if np.sum(B1) == 0:
                for idx_2, line_2 in enumerate(l_lines[idx + 1 :]):
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B1[int(slot)] = 1
                    else:
                        break

            # Same with B2
            elif np.sum(B2) == 0:
                for idx_2, line_2 in enumerate(l_lines[idx + 1 :]):
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B2[int(slot)] = 1
                    else:
                        break
            else:
                break

    data_json = {"beam1": [int(ii) for ii in B1], "beam2": [int(ii) for ii in B2]}

    with open(filling_scheme_path.split(".json")[0] + "_converted.json", "w") as file_bool:
        json.dump(data_json, file_bool)
    return B1, B2


def load_and_check_filling_scheme(filling_scheme_path):
    if not filling_scheme_path.endswith(".json"):
        raise ValueError("Filling scheme must be in json format")

    with open(filling_scheme_path, "r") as fid:
        d_filling_scheme = json.load(fid)

    if "beam1" in d_filling_scheme.keys() and "beam2" in d_filling_scheme.keys():
        # If the filling scheme not already in the correct format, convert
        if "schemebeam1" in d_filling_scheme.keys() or "schemebeam2" in d_filling_scheme.keys():
            d_filling_scheme["beam1"] = d_filling_scheme["schemebeam1"]
            d_filling_scheme["beam2"] = d_filling_scheme["schemebeam2"]
            # Delete all the other keys
            d_filling_scheme = {
                k: v for k, v in d_filling_scheme.items() if k in ["beam1", "beam2"]
            }
            # Dump the dictionary back to the file
            filling_scheme_path = filling_scheme_path.replace(".json", "_converted.json")
            with open(filling_scheme_path, "w") as fid:
                json.dump(d_filling_scheme, fid)

            # Else, do nothing

    else:
        # One can potentially use b1_array, b2_array to scan the bunches later
        b1_array, b2_array = reformat_filling_scheme_from_lpc(filling_scheme_path)
        filling_scheme_path = filling_scheme_path.replace(".json", "_converted.json")

    return filling_scheme_path
