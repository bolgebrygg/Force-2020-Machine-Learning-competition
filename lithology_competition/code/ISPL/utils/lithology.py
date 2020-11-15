"""
Lithologies maps.

Nomenclature:
- code: code associated to the lithology
- label: label used for classification task. Used in the custom scoring function
- name: human-readable name for the litology
"""

"""
Mapping from code to classification label.
As given by the challenge
"""
code2label = {
    30000: 0,
    65030: 1,
    65000: 2,
    80000: 3,
    74000: 4,
    70000: 5,
    70032: 6,
    88000: 7,
    86000: 8,
    99000: 9,
    90000: 10,
    93000: 11
}

"""
Mapping from code to name.
As given by the challenge
"""
code2name = {
    30000: 'Sandstone',
    65030: 'Sandstone/Shale',
    65000: 'Shale',
    80000: 'Marl',
    74000: 'Dolomite',
    70000: 'Limestone',
    70032: 'Chalk',
    88000: 'Halite',
    86000: 'Anhydrite',
    99000: 'Tuff',
    90000: 'Coal',
    93000: 'Basement',
}

"""
Mapping from classification label to code.
"""
label2code = dict((v, k) for k, v in code2label.items())

"""
Mapping from classification label to name.
"""
label2name = dict((k, code2name[v]) for k, v in label2code.items())
