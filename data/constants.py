class_names = {
    "TuSimple": [
        "no lane",
        "continuous yellow",
        "continuous white",
        "dashed",
        "double dashed",
        "Botts' dots",
        "double continuous yellow",
        "unknown"
    ],
    "WATO": [
        "none",
        "dashed white",
        "dashed yellow",
        "solid white",
        "solid yellow",
        "double dashed white",
        "double dashed yellow",
        "double solid white",
        "double solid yellow",
        "solid dashed white",
        "solid dashed yellow",
        "dashed solid white",
        "dashed solid yellow",
        "botts dots",
        "curb"
    ]
}

display_names = {
    "TuSimple": [
        "none",
        "cont-y",
        "cont-w",
        "dash",
        "doub-dash",
        "botts",
        "doub-cont-y",
        "unknown"
    ],
    "WATO": [
        "none",
        "dash-w",
        "dash-y",
        "sol-w",
        "sol-y",
        "doub-dash-w",
        "doub-dash-y",
        "doub-sol-w",
        "doub-sol-y",
        "sol-dash-w",
        "sol-dash-y",
        "dash-sol-w",
        "dash-sol-y",
        "botts",
        "curb"
    ]
}

wato2tusimple_class_mapping = [
    0,  # no lane
    3,  # dashed
    3,  # dashed
    2,  # continuous white
    1,  # continuous yellow
    4,  # double dashed
    4,  # double dashed
    6,  # double continuous yellow
    6,  # double continuous yellow
    4,  # double dashed
    4,  # double dashed
    4,  # double dashed
    4,  # double dashed
    5,  # Botts' dots
    7  # unknown
]
