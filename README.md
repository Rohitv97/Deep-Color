# Deep-Color
Colorization of Images using CNN and GAN.

* Go through the Dependencies folder to install whatever is needed

* to_Gray python script can be used to convert a color image to Grayscale

* Following are the Dependencies to be installed: 

# Instructions

## Setting up Dependencies

## Installing OpenCV
* Run the following commands
  ```
  pip install opencv-contrib-python
  ```
  
## Custom installting all dependencies for GPU

### Installing Cuda-9.0
* Download the cuda runfile [here](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)
* Install some other dependencies and then install cuda

  ```
  sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
  sudo chmod 777 *.runfile  # '*.runfile' denotes the file name you just downloaded
  sudo ./cuda_9.0.176_384.81_linux.run -toolkit -samples -silent -override #
  ```
* Create a symbolic link to cuda to avoid missing library errors

  ```
  cd /usr/local
  sudo ln -s /usr/local/cuda-9.0 cuda
  ```
* Lower the gcc version of the system to before 6

  ```
  gcc --version # check the gcc version
  sudo apt install gcc-5 g++-5
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
  ```

* Set environment variables

  ```
   export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
  ```
* Also modify and add path to .bashrc just in case

  ```
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
  ```
* Verify cuda-9.0 installation

  ```
  cd NVIDIA_CUDA-9.0_Samples/5_Simulations/fluidsGL
  make clean && make
  ./fluidsGL
  ```
  If cuda-9.0 has been installed properly, there should be no error messages during making. After running you will the fluid window.

### Installing cuDNN
* Go to this [page](https://developer.nvidia.com/rdp/cudnn-archive) and create an account
* Click “Download cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0” and download the following files: runtime library, developer library, and code samples and user guide.
* Run the following commands to install

  ```
  sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
  sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
  sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
  ```
* Install Freeimage and verify cuDNN

  ```
  sudo apt-get install libfreeimage3 libfreeimage-dev
  cp -r /usr/src/cudnn_samples_v7/ $HOME
  cd $HOME/cudnn_samples_v7/mnistCUDNN
  make clean && make
  ./mnistCUDNN
  ```

### Installing Tensorflow
* We'll need to setup a virtual environment and then install.

  ```
  sudo apt-get install libcupti-dev
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

  sudo apt-get install python3-pip python3-dev python-virtualenv
  virtualenv --system-site-packages -p python3 tensorflow # create a enviroment named tensorflow
  ```
* Installing TensorFlow CPU

  ```
  source ~/tensorflow/bin/activate
  pip3 install --upgrade tensorflow  # install the cpu version
  ```
* Verifying TensorFlow CPU
  Make sure you are in the same TF environment. Enter python

  ```
  import tensorflow as tf
  hello = tf.constant('Hello, TensorFlow!')
  sess = tf.Session()
  print(sess.run(hello))
  ```

* Installing TensorFlow GPU
  Make sure you are in the same environment

  ```
  pip3 install --upgrade tensorflow-gpu
  ```
  Verify as before

### Install PyTorch
* First we need to install Anaconda. Click [here](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh) to download the file.

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

### Installing other dependencies
* Installing Scikit,Numpy,Pandas and other libraries. Make sure pip has been installed in the system.

  ```
  pip3 install numpy
  pip3 install pandas
  pip3 install scipy
  pip3 install scikit-learn
  pip3 install matplotlib
  ```


### Install Keras with backend as TensorFlow
* Make sure Tensorflow environment is activate

  ```
  pip install -q keras
  pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
  pip install torchvision
  sudo apt-get -qq install -y graphviz
  pip install -q pydot
  pip install mxnet-cu90
  pip install scikit-image
  ```

* Verify installation, by running them in Ipython or some Python script and check version.

  ```
  ipython
  import tensorflow as tf
  import torch
  import cv2
  import mxnet
  import keras
  ```






## Installing Anaconda and running Project

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
