# Deep-Color
Colorization of Images using CNN and GAN.

* Go through the Dependencies folder to install whatever is needed

* to_Gray python script can be used to convert a color image to Grayscale

* Following are the Dependencies to be installed: 

# Instructions

## Setting up Dependencies

### Installing OpenCV
* Run the following commands
  ```
  pip install opencv-contrib-python
  ```
### Installing Anaconda

* Click [here](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh) to download the file.

  ```
  bash ~/Downloads/'file which was just downloaded'
  ```

  Agree to the TnC. Let it set Path automatically. Wait for installations to finish.
  Install VSCode if you want, else you can skip.
  Run the following:

  ```
  source ~/.bashrc
  ```

  In another teminal run:
  ```
  anaconda-navigator
  ```
  If Navigator opens up, Anaconda has been installed successfully.

* Do the following steps
  Check your group by typing this
  ```
  groups
  ```
  The first group is usally the group right now in use
  Also keep your username in mind. And now execute the following command:
  ```
  chown -R YOUR_group:YOUR_USER_name anaconda3
  ```
* Set channels, and download pytorch from mirror link (faster)

  ```
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  conda config --set show_channel_urls yes
  conda install pytorch torchvision cuda91 -c https://mirrors.ustc.edu.cn/anaconda/cloud/pytorch/
  ```
  Open a python script and try to import torch. If it imports, you've installed PyTorch
  
### Installing Tensorflow, Keras and Scikit-Image
 
  ```
  conda install -c anaconda keras 
  conda install -c anaconda scikit-image
  ```
