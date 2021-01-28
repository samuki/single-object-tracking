# Evalutating Single Object Tracking For Autonomous Driving

<p align="center">
<img src="images/boxes.png" width="250" />
</p>

In this project, I applied and evaluated Single Object Tracking by using the object tracking system [SiamMask](https://github.com/foolwood/SiamMask) on the Audi Autonomous Driving Dataset [A2D2](https://www.a2d2.audi/a2d2/en.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/).  
The goal was to test, how applicable SiamMask to the task of tracking individual vehicles in those datasets without explicitly fine-tuning it.

The tracker can be evaluated by using the semantic segmentation and instance segmentation images which were provided by human annotators: 

<p align="center">
<img src="images/20181107133258_camera_frontcenter_000000250.png" width="250" /> <img src="images/20181107133258_instance_frontcenter_000000250.png" width="250" /> <img src="images/20181107133258_label_frontcenter_000000250.png" width="250" />
</p>

In the A2D2, only a subset of the video frames is annotated with segmentations.
This leads to fail cases where the tracker switches to other objects, if the gap between the scene displayed in two consequtive frames is too big:  

<p align="center">
<img src="images/tracker_jumps.png" width="250" /> <img src="images/tracker_jumps2.png" width="250" /> <img src="images/tracker_jumps3.png" width="250" />
<img src="images/mf2.png" width="250" /> <img src="images/mf3.png" width="250" /> <img src="images/mf4.png" width="250" />
</p>

For details see the whole [report](Kiegeland_Project_Report.pdf)

<p align="center">
<img src="images/8.png" width="250" /> <img src="images/9.png" width="250" /> <img src="images/10.png" width="250" />
</p>

How to use the GUI: 

python GUI.py --config "../SiamMask/experiments/siammask_sharp/config_davis.json" --resume "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth" --dataset "path/to/dataset" --object_lookup "path/to/lookup"
