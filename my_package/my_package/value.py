DIM = {
    "CONCH": 512,
    "CONCH_512": 512,
    "UNI": 1024,
    "Virchow2": 2560,
    "gigapath": 1536,
    "UNI2": 1536,
    "hibou_l": 1024,
    "hibou_b": 768,
    "phikon2": 1024
}

FLPOPS = {
    "phikon2": 59.685687296,
    "hibou_b": 22.30359552,
    "hibou_l": 79.12851456,
    "CONCH": 67.196417024,
    "CONCH_512": 16.799104256,
    "gigapath": 223.424280576,
    "UNI": 59.685687296,
    "UNI2": 180.372461568,
    "Virchow2": 164.57206272,
}

PARAMS = {
    "phikon2": 0.30309888,
    "hibou_b": 0.08552064,
    "nibou_l": 0.3033408,
    "CONCH": 0.085647616,
    "CONCH_512": 0.085647616,
    "gigapath": 1.134526976,
    "UNI": 0.30309888,
    "UNI2": 0.680913408,
    "Virchow2": 0.630817024,
}

MIL_FLPOPS = {
    "S4": 2.625542144,
    "transmil": 3.039568896,
}

MIL_PARAMS = {
    "S4": 0.00263271,
    "transmil": 0.002411542,
}

if __name__ == "__main__":
    # total = 0
    for key, value in FLPOPS.items():
        # total += i# + MIL_FLPOPS["transmil"]
        print(f"{key}: {value+MIL_FLPOPS["transmil"]}")
    # print(total)