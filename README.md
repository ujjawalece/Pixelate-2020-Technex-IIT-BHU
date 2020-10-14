# Pixelate_2020_Technex_IIT-BHU
Here we are providing the python and arduino code which we used in **Pixelate 20**, an image processsing event. It is organised under **Technex**, an annual Technical fest of **IIT(BHU)**.

### Problem statement-

Complete problem statement is [here](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/PS%20Pixelate.pdf).
Basically we have to design a BOT which can detect different shape and colour and can find an optimal path described in [PS](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/PS%20Pixelate.pdf). And can carry boxex from one point to another by using that path.

### Solution-
* Here we are attaching the feed which our bot gets from the camera placed above the arena.
![arena](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/Topview_of_arena.jpg)

* We used OpenCV for shape detection and colour detection. A mask image of yellow coloured object is here-
![mask image](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/Mask_image.png)

* Then find there coordinates and make a 9X9 matrix from these shapes. Here is an image where the black and white dots showing the 81 coordinates of coloured object which are stored in a 9X9 matrix-
![img](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/img.png)

* Then an optimal path was plotted to get the optimal trajectory(we used dijkstra theorem here).

* Once we have path,we know the coodinates of source and destination and also we used **Aruco markers** to find the coordinates of our bot. Here is an image of 5X5 aruco-
![aruco](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/Aruco_5x5.jpg)

### Video-

* [Here](https://drive.google.com/file/d/1IY2eWKTDYjUKQptkgh5_7EyUIm-fnXS6/view?usp=sharing) is the video of our working bot.

### Codes-

* [Python code](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/Python.py)
* [Arduino code](https://github.com/ujjawalece/Pixelate_2020_Technex_IIT-BHU-/blob/master/Arduino.ino)
