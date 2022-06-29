# **Introduction: Satellite 2 Map**
This project implementation consists of the following methods:
1. Standard GPU based - "satellite2map.py" script trains a discriminator model that utilizes multiple GPUs to accurately map satellite imagery to simpler maps. 
2. The generated image is then compared to the actual mapping, and the model adjusts itself accordingly.

### **Software requirements**
1. Python 3.7 or above
2. Visual Studio Code or another Python IDE
3. Download [maps_256.npz](https://drive.google.com/file/d/1B-69QY4RIUHgJk0L-TeU761N222xd8O8/view?usp=sharing)

### **Standard GPU-based implementation**

###### **Clone the repository**

###### **Set up the virtual environment**
1. Create a new folder and name it as Satellite2Map.
2. Copy satellite2map.py from cloned repository and place it inside Satellite2Map folder.
3. In Visual Studio Code, open satellite2map.py in preparation to run the script. 

####  **Set up the  conda environment**
        conda create -n  satellite2map python=3.8
        conda activate satellite2map

###### **Install python packages**
      
            pip install -r requirements.txt      

###### **Download the data**
1. Place the downloaded "maps_256.npz" file inside Satellite2Map folder
2. Rename the path in chunk 3 according to the new path location.

## **Run the model**
Now, we are all set to run the script.
 python satellite2map.py
## **Walkthrough Video**
**Implementation**\
[<img src="https://github.com/stccenter/Satellite2Map/blob/main/Images/Videos.jpg" width="60%">](https://youtu.be/W8YkFw8Y_ew)
