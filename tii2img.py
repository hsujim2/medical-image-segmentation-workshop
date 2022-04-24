from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

#set PATH first, in windows, change "\" to "\\". ex:D:\\userdata\\...
file_path = 'E:\\userdata\\Desktop\\competition'

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        #print("on_created", event.src_path.split('.')[1])
        if event.src_path.split('.')[1] == "nii":
            # 讀取 影像
            itk_image = sitk.ReadImage(event.src_path)  # Read file
            image_arr = sitk.GetArrayFromImage(itk_image)  # Get raw data
            print("image_arr shape =", image_arr.shape)
            # 轉存為 PNG 圖檔
            out_file = event.src_path.split('.')[0] + ".png"
            plt.imsave(out_file, -image_arr[0], cmap='Greys')
            print("create new file:",out_file)

event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path=file_path, recursive=False)
observer.start()
while True:#死迴圈
    try:
        pass
    except KeyboardInterrupt:
        observer.stop()