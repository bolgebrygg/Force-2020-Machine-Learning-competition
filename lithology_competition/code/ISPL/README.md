# FORCE challenge - ISPL Team submission package
Image and Sound Processing Lab @ Politecnico di Milano

## Authors
- Luca Bondi (luca.bondi@polimi.it)
- Vincenzo Lipari (vincenzo.lipari@polimi.it)
- Paolo Bestagini (paolo.bestagini@polimi.it)
- Francesco Picetti (francesco.picetti@polimi.it)
- Edoardo Daniele Cannas (edoardo.daniele.cannas@polimi.it)

## Requirements
- GPU
- Conda

## Environment setup
```bash
conda env create -f environment.yaml
conda activate force-ispl
```

## Test
```
python3 -m unittest
```

## Run
```bash
python3 main.py <PATH TO WELL DATA CSV> <PATH TO SUBMISSION CSV>
```
