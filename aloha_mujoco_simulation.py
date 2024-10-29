# imports
import mujoco
import numpy as np
import cv2 
import zarr 



# defining xml paths for the models. 


xml_aloha2 = 'aloha2/sim_cube_transfer.xml'

# robot arm trajectory data
qpos_zarr = zarr.open('actions.zarr',mode='r')
qpos_data = qpos_zarr['action']

# defining the model 
model = mujoco.MjModel.from_xml_path(xml_aloha2)
data = mujoco.MjData(model)


# process trajectory data to fit data.qpos
aloha2_gripper_value = lambda x: 0.0076 + (0.041 - 0.0076)*x 
def set_action(data,action):
        left_gripper,right_gripper = action[6],action[13]

        new_action = np.zeros((16))
        
        # updated left arm position 
        new_action[0:6] = action[0:6]
        new_action[6:8] = [aloha2_gripper_value(left_gripper)]*2
        
        # updated right gripper positioon
        new_action[8:14] = action[7:13]
        new_action[14:16] = [aloha2_gripper_value(right_gripper)]*2

        data.qpos[0:16] = new_action



#simulation
height,width= 480,640
frames = []
frame_rate = 30
max_timesteps = len(qpos_data)


with mujoco.Renderer(model, height,width) as renderer:
  for i in range(len(qpos_data)):
    while data.time < i/frame_rate:
        
        mujoco.mj_step(model,data)

        set_action(data,qpos_data[i])
        
    
    renderer.update_scene(data,camera='teleoperator_pov')
  
    img = renderer.render()
    frames.append(img)


# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('visualize.avi', fourcc, fps=frame_rate ,frameSize=(width, height), isColor= True)
for frame in frames:
    frame =cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
out.release()

