# Requirements:
mujoco 3.1.1 or later
numpy 
cv2
zarr

# overview
This is a simplified version of real - sim teleoperation on ALOHA2. Real robot trajectory data is stored in the "actions.zarr" file and is used to update model qpos data during each step of the simulation. And the visualization of the simulation is stored in "visualize.avi".