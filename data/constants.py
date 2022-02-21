# Copyright (C) 2021 Anita Hu, Sinclair Hudson, Martin Ethier.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

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
    },
    "2Class": {
        0: "no lane",
        1: "dashed",
        2: "continuous",
    },
    "3Class": {
        0: "no lane",
        1: "dashed",
        2: "continuous",
        3: "double dashed",
    },
}

wato_2class_mapping = {
    0: 0,  # no lane
    1: 1,  # dashed white
    2: 1,  # dashed yellow
    3: 2,  # continuous white
    4: 2,  # continuous yellow
    5: 1,  # double dashed white
    6: 1,  # double dashed yellow
    7: 2,  # double solid white
    8: 2,  # double solid yellow
    9: 1,  # solid dashed white
    10: 1,  # solid dashed yellow
    11: 2,  # dashed solid white
    12: 2,  # dashed solid yellow
    13: 1,  # botts dots
    14: 2  # curb
}

tusimple_2class_mapping = {
    0: 0,  # no lane
    1: 2,  # continuous yellow
    2: 2,  # continuous white
    3: 1,  # dashed
    4: 1,  # double dashed
    5: 1,  # botts dots
    6: 2,  # double continuous yellow
    7: 0  # unknown
}

wato_3class_mapping = {
    0: 0,  # no lane
    1: 1,  # dashed white
    2: 1,  # dashed yellow
    3: 2,  # continuous white
    4: 2,  # continuous yellow
    5: 3,  # double dashed white
    6: 3,  # double dashed yellow
    7: 2,  # double solid white
    8: 2,  # double solid yellow
    9: 1,  # solid dashed white
    10: 1,  # solid dashed yellow
    11: 2,  # dashed solid white
    12: 2,  # dashed solid yellow
    13: 1,  # botts dots
    14: 2  # curb
}

tusimple_3class_mapping = {
    0: 0,  # no lane
    1: 2,  # continuous yellow
    2: 2,  # continuous white
    3: 1,  # dashed
    4: 3,  # double dashed
    5: 1,  # botts dots
    6: 2,  # double continuous yellow
    7: 0  # unknown
}
