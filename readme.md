# FORCE 2020 Machine Learning Competition

For citation please use: Bormann P., Aursand P., Dilib F., Dischington P., Manral S. 2020. FORCE Machine Learning Competition. https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition

**Link to contest https://www.npd.no/en/force/events/machine-learning-contest-with-wells-and-seismic/**

![Sponsors](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/blob/master/bottom-sponsor-6.jpg)

## Lithology prediction

The objective of the lithology prediction competition was to correctly predict lithology labels for provided well logs, provided NPD lithostratigraphy and well X, Y position.

The competition was scored using a penalty matrix. Some label mistakes are penalized more than others, see starter notebook and penalty matrix for details.

All datasets used in the competition and the starter notebook can be found under `lithology_competition/data`

**Petrel ready files and standard well log las files, all csv file data, predictions and submited modeles and weights** can also be found here along with a host of other free geosience subsurface data. The folder is called **FORCE 2020 lithofacies prediction from well logs competition** (https://drive.google.com/drive/folders/0B7brcf-eGK8CRUhfRW9rSG91bW8) 

### Results of final scoring

A total of 329 teams signed up for the competition and 148 teams submitted predictions on the open test dataset to enter the competition leaderboard. At the end of the competition the top 30 teams in the leaderboard were invited to submit their pre-trained models for scoring on a hidden dataset. Of these teams 13 submitted code that was easily runnable by the organizers, giving the final scores below.  

### Description and analysis of the results
**Writeup geological and organisational summary of the results** (https://docs.google.com/document/d/13XAftsBVHIm01ZN0lP56Q4hZ9hgdYR1G_6KeV2DdzOA/edit?usp=sharing) <br/>
**Writeup Technical summary of the results** (https://docs.google.com/document/d/1cQc9zzQJIeEC4JC8y-6hvVPDarflpb9pjWNIrQcWyWg/edit)

| Team | Leaderboard score | Leaderboard rank | Final test score | Final rank |
|---|---|---|---|---|
| Olawale Ibrahim | -0.5118 | 24 | -0.4690 | 1 |
| GIR Team | -0.5037 | 11 | -0.4792 | 2 |
| Lab.ICA-Team / Smith A. | -0.4943 | 6 | -0.4954 | 3 |
| H3G (Haoyuan Zhang, Harry Brandsen, Gregory Barrere, Helena Nandi Formentin) | -0.509 | 17 | -0.5045 | 4 |
| ISPL Team | -0.4885 | 2 | -0.5084 | 5 |
| Jiampiers C. | -0.5014 | 9 | -0.5087 | 6 |
| José Bermúdez | -0.5052 | 14 | -0.5091 | 7 |
| Bohdan Pavlyshenko | -0.5112 | 22 | -0.5171 | 8 |
| Jeremy Zhao | -0.5264 | 31 | -0.5173 | 9 |
| Campbell Hutcheson | -0.505 | 13 | -0.5221 | 10 |
| David P. | -0.4775 | 1 | -0.5256 | 11 |
| SoftServe Team | -0.4936 | 3 | -0.5263 | 12 |
| Dapo Awolayo | -0.5121 | 25 | -0.9441 | 13 |


## Mapping faults on seismic FORCE 2020 competition

A total of 80 teams signed up for the competition but only 5 submitted a valid scored fault cube in the end. We were a bit surprised by the low rate of submissions. It is most liley explained by the fact that the seismic dataset was not one of the shining examples where ML fault models perform wonderfulle but more of a standard seismic cube with some migration issues.

**Sparveon won the competition followed by Equinor and Woodside**

A geological summary write up of the competition can be found here: https://docs.google.com/document/d/1DURjbg2o5C4N5QUaK0bsUGHUb6QLKdhGuKjpZqTkm9g/edit?usp=sharing  <br/>
A slidepack comparing the results can be found here: https://drive.google.com/file/d/1r0k4MU22QmhsxgZ7BJZ8sYveDRQwJlne/view?usp=sharing <br/>
All the submited scored blind cubes and the samples of the training data can be accesssed here: https://drive.google.com/drive/folders/1Hu4VJN9xLOWixSMdf2xN6fRk0zmOz-1J?usp=sharing<br/>


