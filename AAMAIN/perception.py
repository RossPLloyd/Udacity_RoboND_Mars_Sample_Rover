import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
	# Create an array of zeros same xy size as img, but single channel
	color_select = np.zeros_like(img[:,:,0])
	# Require that each pixel be above all three threshold values in RGB
	# above_thresh will now contain a boolean array with "True"
	# where threshold was met
	above_thresh = (img[:,:,0] > rgb_thresh[0]) \
				& (img[:,:,1] > rgb_thresh[1]) \
				& (img[:,:,2] > rgb_thresh[2])
	# Index the array of zeros with the boolean array and set to 1
	color_select[above_thresh] = 1
	# Return the binary image
	return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
	# Identify nonzero pixels
	ypos, xpos = binary_img.nonzero()
	# Calculate pixel positions with reference to the rover position being at the 
	# center bottom of the image.  
	x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
	y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
	return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
	# Convert (x_pixel, y_pixel) to (distance, angle) 
	# in polar coordinates in rover space
	# Calculate distance to each pixel
	dist = np.sqrt(x_pixel**2 + y_pixel**2)
	# Calculate angle away from vertical for each pixel
	angles = np.arctan2(y_pixel, x_pixel)
	return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
	# Convert yaw to radians
	yaw_rad = yaw * np.pi / 180
	xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
							
	ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
	# Return the result  
	return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
	# Apply a scaling and a translation
	xpix_translated = (xpix_rot / scale) + xpos
	ypix_translated = (ypix_rot / scale) + ypos
	# Return the result  
	return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
	# Apply rotation
	xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
	# Apply translation
	xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
	# Perform rotation, translation and clipping all at once
	x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
	y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
	# Return the result
	return x_pix_world, y_pix_world

# Define a function to perform a perspective transform. This has been updated with code that applies a mask
# in order to prevent the "out of bounds" camera region from being used in the update of the map.
def perspect_transform(img, src, dst):           
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
	mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
	return warped, mask



#define a function that thresholds for the yellow rocks, and populates an empty array of zeros
#with 1's
def find_rocks(img, levels=(110, 110, 50)):
	rockpix = ((img[:,:,0] > levels[0])) \
			& (img[:,:,1] > levels[1]) \
			& (img[:,:,2] < levels[2])
		
	color_select = np.zeros_like(img[:,:,0])
	color_select[rockpix] = 1
	
	return color_select

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
	#define a count to keep track of time steps. Ultimately I could not apply this as I wanted,
	#which was to define a 'stuck' routine, however it is left here for illustration. The idea is that
	#if the rover position has not changed in 10 seconds, it is stuck!
	#nb the code for this in the decision step has been removed as it was interfering with other steps
	Rover.count += 1
	#multiply the first and last indexed xposition and yposition together and take the absolute
	Rover.xypos_mult = abs((Rover.pos[0] * Rover.pos[1]))
	#this populates the empty list from beginning to end, and then shifts the data to the right once
	#the array size is 450. At 45fps this equates to 10 seconds of data.
	if (len(Rover.buffer_xypos) < 450):
		Rover.buffer_xypos.insert(0, Rover.xypos_mult)
	elif (len(Rover.buffer_xypos) == 450):
		Rover.buffer_xypos.pop()
		Rover.buffer_xypos.insert(0, Rover.xypos_mult)
	image = Rover.img

	# Define calibration box in source (actual) and destination (desired) coordinates
	# These source and destination points are defined to warp the img
	# to a grid where each 10x10 pixel square represents 1 square meter
	# The destination box will be 2*dst_size on each side
	dst_size = 5 
	# Set a bottom offset to account for the fact that the bottom of the img 
	# is not the position of the rover but a bit in front of it
	# this is just a rough guess, feel free to change it!
	bottom_offset = 6
	source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
	destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset], [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset], [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], [Rover.img.shape[1]/2 - dst_size,Rover.img.shape[0] - 2*dst_size - bottom_offset],])
	#DONE: 1- 5 Perspective Transform and mask
	warped, mask = perspect_transform(Rover.img, source, destination)
	#Apply colour theshold to identify navigable terrin / obstacles / rock samples
	threshed = color_thresh(warped)
	xpix, ypix = rover_coords(threshed)
	dist, angles = to_polar_coords(xpix, ypix)
	Rover.nav_angles = angles

	obs_map = np.absolute(np.float32(threshed) - 1) * mask

	#add new line regarding rover.vision_image to inject rover cam view
	Rover.vision_image[:,:,2] = threshed * 255
	Rover.vision_image[:,:,0] = obs_map * 255
	

	
	#convert rover-centric pixel values to world co-ordinates
	world_size = Rover.worldmap.shape[0]
	scale = 2 * dst_size

	x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
	obsxpix, obsypix = rover_coords(obs_map)
	obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
		
		
	#define some variables for measuring roll and pitch
	pitch_value = Rover.pitch
	roll_value = Rover.roll

	# In order to improve mapping fidelity, only map when pitch_value < 0.01 or 350 >= pitch_value <=360:  
	#Update Worldmap
	
	if ((pitch_value < 0.04) and ((roll_value < 0.04) or (roll_value > 359))):
		Rover.worldmap[y_world, x_world, 2] = 255
	elif ((pitch_value > 358) and ((roll_value < 0.05) or (roll_value > 359))):
		Rover.worldmap[y_world, x_world, 2] = 255
	else: 
		pass
	
	if ((pitch_value < 0.04) and ((roll_value < 0.04) or (roll_value > 359))):
		Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
	elif ((pitch_value > 358) and ((roll_value < 0.05) or (roll_value > 359))):
		Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
	else:
		pass
	
	
	
	#Finding rocks. This updates the map with the position of rocks
	rock_map = find_rocks(warped, levels=(110, 110, 50))
	if rock_map.any():
		rock_x, rock_y = rover_coords(rock_map)
		rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
		rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
		rock_idx = np.argmin(rock_dist)
		rock_xcen = rock_x_world[rock_idx]
		rock_ycen = rock_y_world[rock_idx]

		Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
		Rover.vision_image[:, :, 1] = rock_map * 255
	else:
		Rover.vision_image[:, :, 1] = 0
		
	return Rover