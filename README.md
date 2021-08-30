### **Introduction**
This git repository is created for the satellite mapping project. This project is implemented in this method:
1. Standard GPU based - "Translation4_MultiGPU.ipynb" notebook is to train a discriminator model that utilizes multiple GPUs, to accurately map satellite imagery to simpler maps. 

The generated image is then compared to the actual mapping, and the model adjusts itself accordingly.

### **Software requirements**
1. Python 3.7 or above
2. Jupyter notebooks
3. Download [Pix2Pix](https://gmuedu.sharepoint.com/sites/REU-GRP/Shared%20Documents/General/Image%20Mapping/maps_256.npz)

### **Standard CPU-based implementation**

#### **I - Clone the repository**

#### **II - Set up the virtual environment**
1. Create a new folder and name it as Satellite2Map.
2. Copy Translation4_MultiGPU.ipynb from cloned repository and place it inside Satellite2Map folder.
3. Put the Pix2Pix file into your Satellite2Map folder.
4. In Jupyter notebooks, open Translation4_MultiGPU in preparation to run the notebook. 

#### **III - Install python packages**
      
            pip install TensorFlow tensorflow_datasets numpy time random matplotlib       

#### **IV Download the data**
1. Place the downloaded "maps_256.npc" file inside Satellite2Map folder
2. Rename the path in chunk 3 according to the new path location.

#### **V - Run the notebook**
Now, we are all set to run the Jupyter notebook.
