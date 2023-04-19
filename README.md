# Segment_Anything_Model

Clone this repo & cd into it

Create and activate virtual environment.
```
python -m venv .venv
.\.venv\source\activate
```

Upgrade pip
```
python -m pip install --upgrade pip
```

Install requirements
```
pip install \
'git+https://github.com/facebookresearch/segment-anything.git'

pip install -q roboflow supervision
wget -q \
'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
```

Inference on images
```
python SAM.py --source images/horse.jpg --save output.jpg
```

Inference on videos
```
python SAM.py --source test_video.mp4 --save output.mp4
```
