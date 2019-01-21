# fire_trainer
Using simulated aerial fire images to validate classification of real fire images.

# Test Results
The images which appear below were tested with a network trained using only the simulated fire
images generated by the Unity scripts in the Traingen project.  The title of each image indicates
whether the network detected smoke and may be interpreted as follows:


The title of teach image is interpreted as follows:
```
ps = Predicted Smoke
ls = Labeled Smoke
pn = Predicted No Smoke
ln = Labeled No Smoke
```

If the prediction was correct the second character will match, as in: "ps ls" and "pn ln"

The other characters appearing after these contain the last few characters of the file name. 

The images appearing below represent an overall test accuracy of: 0.875

![Page 1](test_results_figure_1.jpg)
![Page 2](test_results_figure_2.jpg)
![Page 3](test_results_figure_3.jpg)

