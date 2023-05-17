# Lightweight learning with label proportions on satellite imagery

Raúl Ramos-Pollán, raul.ramos@udea.edu.co<br/>
Fabio A. González, fagonzalezo@unal.edu.co

# Downloading datasets
The four datasets  in our work with Sentinel2 RGB imagery and different labels are available at Zenodo:

|region |labels| km2| resolution | available at |
|---|---|---|---|---|
|colombia-ne| esaworldcover |69193| 10m| https://zenodo.org/record/7935303|
|colombia-ne |humanpop |69193 |250m |https://zenodo.org/record/7939365|
|benelux |esaworldcover |72213 |10m |https://zenodo.org/record/7935237|
|benelux |humanpop |72213 |250m |https://zenodo.org/record/7939348

# Running the experiments (tranining models)

1. Download the zip file any of the datasets above and unzip, for instance under `/opt/data`
2. Under `scripts` select the script for `esaworldcover` or `humanpop` that you want to run, and check the location of the `DATASET` variable is correct. The `TAG` will be used to report results to `wandb`
3. Have your `wandb` token ready.
4. Run the experiment:

```
cd scripts
sh run_esaworldcover.sh
```

while running, hiting `ctrl-c` once will abort training, but will still loop through the train, val and test datasets to measure and report results to `wandb`

you can also use the Docker files under `docker` to start a container configured with `tensorflow` to run your experiments on a GPU enabled machine.

# Results and figures

The IPython notebooks under `notebooks` contain the code to generate the figures used in the paper (maps, metrics, etc.)