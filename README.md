## YOLOv8 Benchmark for WaterScenes 
---


### Data Preparation
1. Download dataset from [WaterScenes](https://github.com/WaterScenes/WaterScenes)


2. Train the model
   * Modify paths in `train.py`

      ```python
      classes_path    = 'model_data/waterscenes.txt'
      radar_file_path = "/data/WaterScenes_Published/VOCradar640_new"
      ```

   * run `CUDA_VISIBLE_DEVICES=0 python train.py`

3. Test the model. 
   * Download trained weight: [Baidu Netdisk](https://pan.baidu.com/s/1I1DEUHtaNxs4Qul0L5_kag?pwd=85ud)

   * Modify paths in `yolo.py`

      ```python
            "model_path"        : 'model_data/yolov8_waterscenes.pth',
            "classes_path"      : 'model_data/waterscenes.txt',
            "radar_root"        : '/data/WaterScenes_Published/VOCradar640_new',
      ```

   * Modify paths in `get_map.py`

      ```python
         VOCdevkit_path  = '/data/WaterScenes_Published'
      ```

   * run `python get_map.py`


### Acknowledgement
* https://github.com/guanrunwei
* https://github.com/bubbliiiing/yolov8-pytorch
