This is the code for the project:
"Curriculum Learning for Dynamic Robot Locomotion on Challenging Terrains" by Davide Paglieri, MSc student in Computing (AI & ML) at Imperial College London.

Quick overview of the project can be seen below. For more details check the full dissertation [here]()

•Control architecture, is based on PMTG and 

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

To visualise the experiments in video 2 of the appendix, you must install the following dependencies:

pip3 install pybullet
pip3 install absl-py
pip3 install numpy
pip3 install opensimplex

then run

python3 visualise_generalist.py