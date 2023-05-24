def generate_run_sh(node, generation_number):
    python_command = node.root.parameters["generations"][generation_number]["job_executable"]
    return (
        f'source {node.root.parameters["setup_env_script"]}\n'
        f"cd {node.get_abs_path()}\n"
        f"python {python_command} > output.txt 2> error.txt\n"
    )


def generate_run_sh_htc(node, generation_number):
    python_command = node.root.parameters["generations"][generation_number]["job_executable"]
    return (
        f"#!/bin/bash\nsource {node.root.parameters['setup_env_script']}\ncd"
        f" {node.get_abs_path()}\npython {python_command} > output.txt 2> error.txt\nrm -rf final_*"
        " modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log"
        " __pycache__ twiss* errors fc* optics_orbit_at*\n"
    )


# def get_B1_collision_schedule(B1_bunches, B2_bunches, numberOfLRToConsider):
#     """
#     # See https://github.com/PyCOMPLETE/FillingPatterns/blob/5f28d1a99e9a2ef7cc5c171d0cab6679946309e8/fillingpatterns/bbFunctions.py#L233
#     For given two series of booleans which represent bunches and a array that represent long range collisions,
#     this function returns dataframe related to their collisions from perspective of beam 1.

#     ===EXAMPLE===
#     fillingSchemeDF=importData.LHCFillsAggregation(['LHC.BCTFR.A6R4.B%:BUNCH_FILL_PATTERN'],6666, ['FLATTOP'],flag='next')
#     B1_bunches = fillingSchemeDF['LHC.BCTFR.A6R4.B1:BUNCH_FILL_PATTERN'].iloc[0]
#     B2_bunches = fillingSchemeDF['LHC.BCTFR.A6R4.B2:BUNCH_FILL_PATTERN'].iloc[0]
#     B1CollisionScheduleDF(B1_bunches, B2_bunches, 25)

#     """
#     bunch_value = 1.0
#     # Transforming bunches in to boolean array
#     B1_bunches = np.array(B1_bunches) == bunch_value
#     B2_bunches = np.array(B2_bunches) == bunch_value

#     # For debugging
#     # pdb.set_trace()

#     # Get indexes of Bunches
#     B1_bunches_index = np.where(B1_bunches)[0]
#     B2_bunches_index = np.where(B2_bunches)[0]

#     if isinstance(numberOfLRToConsider, int):
#         numberOfLRToConsider = [numberOfLRToConsider, numberOfLRToConsider, numberOfLRToConsider]

#     B1df = pd.DataFrame()

#     for n in B1_bunches_index:
#         # First check for collisions in ALICE

#         # Formula for head on collision in ALICE is
#         # (n + 891) mod 3564 = m
#         # where n is number of bunch in B1, and m is number of bunch in B2

#         # Formula for head on collision in ATLAS/CMS is
#         # n = m
#         # where n is number of bunch in B1, and m is number of bunch in B2

#         # Formula for head on collision in LHCb is
#         # (n + 2670) mod 3564 = m
#         # where n is number of bunch in B1, and m is number of bunch in B2

#         head_on_names = ["HO partner in ALICE", "HO partner in ATLAS/CMS", "HO partner in LHCB"]
#         secondary_names = ["# of LR in ALICE", "# of LR in ATLAS/CMS", "# of LR in LHCB"]
#         encounters_names = [
#             "BB partners in ALICE",
#             "BB partners in ATLAS/CMS",
#             "BB partners in LHCB",
#         ]
#         positions_names = ["Positions in ALICE", "Positions in ATLAS/CMS", "Positions in LHCB"]

#         colide_factor_list = [891, 0, 2670]
#         number_of_bunches = 3564

#         # i == 0 for ALICE
#         # i == 1 for ATLAS and CMS
#         # i == 2 for LHCB

#         dictonary = {}

#         for i in range(0, 3):
#             collide_factor = colide_factor_list[i]
#             m = (n + collide_factor) % number_of_bunches

#             # pdb.set_trace()
#             # if this Bunch is true, than there is head on collision
#             if B2_bunches[m]:
#                 head_on = m
#             else:
#                 head_on = np.nan

#             ## Check if beam 2 has bunches in range  m - numberOfLRToConsider to m + numberOfLRToConsider
#             ## Also have to check if bunches wrap around from 3563 to 0 or vice versa

#             bunches_ineraction_temp = np.array([])
#             encounters = np.array([])
#             positions = np.array([])

#             first_to_consider = m - numberOfLRToConsider[i]
#             last_to_consider = m + numberOfLRToConsider[i] + 1

#             numb_of_long_range = 0

#             if first_to_consider < 0:
#                 bunches_ineraction_partial = np.where(
#                     B2_bunches[(number_of_bunches + first_to_consider) : (number_of_bunches)]
#                 )[0]

#                 # This represents the absolute position of the bunches
#                 encounters = np.append(
#                     encounters, number_of_bunches + first_to_consider + bunches_ineraction_partial
#                 )

#                 # This represents the relative position to the head-on bunch
#                 positions = np.append(positions, first_to_consider + bunches_ineraction_partial)

#                 # Set this varibale so later the normal syntax wihtout the wrap around checking can be used
#                 first_to_consider = 0

#             if last_to_consider > number_of_bunches:
#                 bunches_ineraction_partial = np.where(
#                     B2_bunches[0 : last_to_consider - number_of_bunches]
#                 )[0]

#                 # This represents the absolute position of the bunches
#                 encounters = np.append(encounters, bunches_ineraction_partial)

#                 # This represents the relative position to the head-on bunch
#                 positions = np.append(positions, number_of_bunches - m + bunches_ineraction_partial)

#                 last_to_consider = number_of_bunches

#             bunches_ineraction_partial = np.append(
#                 bunches_ineraction_temp, np.where(B2_bunches[first_to_consider:last_to_consider])[0]
#             )

#             # This represents the absolute position of the bunches
#             encounters = np.append(encounters, first_to_consider + bunches_ineraction_partial)

#             # This represents the relative position to the head-on bunch
#             positions = np.append(positions, bunches_ineraction_partial - (m - first_to_consider))

#             # Substract head on collision from number of secondary collisions
#             numb_of_long_range = len(positions) - int(B2_bunches[m])

#             dictonary.update(
#                 {
#                     head_on_names[i]: {n: head_on},
#                     secondary_names[i]: {n: numb_of_long_range},
#                     encounters_names[i]: {n: encounters},
#                     positions_names[i]: {n: positions},
#                 }
#             )

#         B1df = pd.concat([B1df, pd.DataFrame(dictonary)])

#     return B1df
