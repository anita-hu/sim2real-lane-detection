classes = {
    "TuSimple": {
        0: "no lane",
        1: "continuous yellow",
        2: "continuous white",
        3: "dashed",
        4: "double dashed",
        5: "Botts' dots",
        6: "double continuous yellow",
        7: "unknown"
    },
    "WATO": {
        0: "none",
        1: "dashed white",
        2: "dashed yellow",
        3: "solid white",
        4: "solid yellow",
        5: "double dashed white",
        6: "double dashed yellow",
        7: "double solid white",
        8: "double solid yellow",
        9: "solid dashed white",
        10: "solid dashed yellow",
        11: "dashed solid white",
        12: "dashed solid yellow",
        13: "botts dots",
        14: "curb"
    }
}

wato2tusimple_class_mapping = {
    0: 0,  # no lane
    1: 3,  # dashed
    2: 3,  # dashed
    3: 2,  # continuous white
    4: 1,  # continuous yellow
    5: 4,  # double dashed
    6: 4,  # double dashed
    7: 7,  # double continuous yellow
    8: 6,  # double continuous yellow
    9: 4,  # double dashed
    10: 4,  # double dashed
    11: 4,  # double dashed
    12: 4,  # double dashed
    13: 5,  # Botts' dots
    14: 7  # unknown
}
