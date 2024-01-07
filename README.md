# NYCU_RL_2023

## pytorchEnv_setup
install Anaconda3
https://www.anaconda.com/download 

open "anaconda powershell prompt"  //if can'd find -- https://www.zhihu.com/question/41511552
$conda create -n ENV_NAME python=3.9   //the accepted python version for pytorch is 3.8 or 3.9
$conda activate ENV_NAME

$[ENV_NAME] pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  //copy the desired env from https://pytorch.org/
$[ENV_NAME] pip install gym
$[ENV_NAME] pip install gymnasium
$[ENV_NAME] pip install pygame //(optional)
$[ENV_NAME] pip install tensorboard  //(optional)
$[ENV_NAME] pip install packaging
$[ENV_NAME] conda install -c conda-forge box2d-py
$[ENV_NAME] pip install opencv-python


[.\final_project_env] pip install -e .