Code for the MSc individual project "Open-Ended Curriculum Learning for Dynamic Robot Locomotion" by Davide Paglieri, at Imperial College London.

Quick overview of the control architecture, reward function, terrain generation and training algorithm can be seen below. Video of the deployed robot can be found [here](https://www.youtube.com/watch?v=IE6DSlAliQ8) and [here](https://www.youtube.com/watch?v=PUApftlqQ_Y). For more details check the full dissertation [here](https://github.com/DavidePaglieri/Open_Ended_CL/blob/main/project_details/final_report.pdf)

<p align="center">
<img style="float: center;" src="project_details/figures/front_page.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/control_architecture.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/reward.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/terrain_generation.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/terrains.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/generalist_open_ended.png" width="665">
</p>

<p align="center">
<img style="float: center;" src="project_details/figures/deployed_robot.png" width="665">
</p>

To visualise the fully blind robot trained with an open-ended curriculum, you must install the following dependencies:

pip3 install pybullet
pip3 install absl-py
pip3 install numpy
pip3 install opensimplex

then run

python3 visualise_generalist.py