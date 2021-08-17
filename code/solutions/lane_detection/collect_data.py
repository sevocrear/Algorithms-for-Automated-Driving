# Code based on Carla examples, which are authored by
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

# How to run:
# Start a Carla simulation
# cd into the parent directory of the 'code' directory and run
# python -m code.solutions.lane_detection.collect_data

import os
from token import LEFTSHIFT
import carla
import random
import pygame
import numpy as np
import cv2
from datetime import datetime


from ...util.carla_util import (
    carla_vec_to_np_array,
    CarlaSyncMode,
    find_weather_presets,
    draw_image,
    get_font,
    should_quit,
)
from .camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    CameraGeometry,
)
from .seg_data_util import mkdir_if_not_exist


store_files = True
town_string = "Town03"
cg = CameraGeometry(height=1.5,pitch_deg = 5, image_width=1920, image_height= 1080, field_of_view_deg= 100)
width = cg.image_width
height = cg.image_height

now = datetime.now()
date_time_string = now.strftime("%m_%d_%Y_%H_%M_%S")


def plot_map(m):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])
    plt.show()


def random_transform_disturbance(transform):
    lateral_noise = np.random.normal(0, 0.3)
    lateral_noise = np.clip(lateral_noise, -0.3, 0.3)

    lateral_direction = transform.get_right_vector()
    x = transform.location.x + lateral_noise * lateral_direction.x
    y = transform.location.y + lateral_noise * lateral_direction.y
    z = transform.location.z + lateral_noise * lateral_direction.z

    yaw_noise = np.random.normal(0, 5)
    yaw_noise = np.clip(yaw_noise, -10, 10)

    pitch = transform.rotation.pitch
    yaw = transform.rotation.yaw + yaw_noise
    roll = transform.rotation.roll

    return carla.Transform(
        carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll)
    )


def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    # print(curvature)
    return np.max(curvature)


def create_lane_lines(
    waypoint, exclude_junctions=True, only_turns=False
):
    # print(str(waypoint.right_lane_marking.type))
    center_list, left_boundary, right_boundary = [], [], []
    for _ in range(60):
        if (
            str(waypoint.right_lane_marking.type)
            + str(waypoint.left_lane_marking.type)
        ).find("NONE") != -1:
            return None, None, None
        # if there is a junction on the path, return None
        if exclude_junctions and waypoint.is_junction:
            return None, None, None
        next_waypoints = waypoint.next(1.0)
        # if there is a branch on the path, return None
        if len(next_waypoints) != 1:
            return None, None, None
        waypoint = next_waypoints[0]
        center = carla_vec_to_np_array(waypoint.transform.location)
        center_list.append(center)
        offset = (
            carla_vec_to_np_array(waypoint.transform.get_right_vector())
            * waypoint.lane_width
            / 2.0
        )
        left_boundary.append(center - offset)
        right_boundary.append(center + offset)

    # EXCLUDE LINES WITH BIG CURVATURE
    # max_curvature = get_curvature(np.array(center_list))
    # if max_curvature > 0.005:
    #     return None, None, None

    # if only_turns and max_curvature < 0.002:
    #     return None, None, None

    return (
        np.array(center_list),
        np.array(left_boundary),
        np.array(right_boundary),
    )


def check_inside_image(pixel_array, width, height):
    noise_flag = 0 # sometimes flying roads occurs (?)
    if np.sum(pixel_array[:,1] < height/10) > 0 :
        noise_flag = 1
    ok = (0 < pixel_array[:, 0]) & (pixel_array[:, 0] < width)
    ok = ok & (0 < pixel_array[:, 1]) & (pixel_array[:, 1] < height)
    ratio = np.sum(ok) / len(pixel_array)
    return ratio > 0.5, noise_flag


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def save_img(image, path, raw=False):
    array = carla_img_to_array(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def save_label_img(lb_left_list, lb_right_list,none_flags_list, path):
    label = np.zeros((height, width, 3))
    colors = [[100, 100, 100], [200, 200, 200]]
    for lb_left, lb_right, idx in zip(lb_left_list, lb_right_list, range(len(none_flags_list))):
        if none_flags_list[idx] == 0:
            for color, lb in zip(colors, [lb_left, lb_right]):
                cv2.polylines(
                    label, np.int32([lb]), isClosed=False, color=color, thickness=5
                )
    label = np.mean(label, axis=2)  # collapse color channels to get gray scale
    cv2.imwrite(path, label)


def get_random_spawn_point(m):
    pose = random.choice(m.get_spawn_points())
    return m.get_waypoint(pose.location)


data_folder = os.path.join("code", "solutions", "lane_detection", "data")

def get_waypoints(waypoints, waypoint, side = "left"):
    if waypoint is not None and side == "left" and (waypoint.lane_change == carla.LaneChange.Left or waypoint.lane_change == carla.LaneChange.Both) and waypoint.lane_type == carla.LaneType.Driving:
        waypoints.append(waypoint.get_left_lane())
    elif waypoint is not None and side == "right" and waypoint.lane_type == carla.LaneType.Driving:
        waypoints.append(waypoint.get_right_lane())
    else:
        return waypoints
    waypoints = get_waypoints(waypoints, waypoints[-1], side = side)
    return waypoints

def main():
    mkdir_if_not_exist(data_folder)
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    client.load_world(town_string)
    world = client.get_world()

    try:
        m = world.get_map()
        # plot_map(m)
        start_pose = random.choice(m.get_spawn_points())
        spawn_waypoint = m.get_waypoint(start_pose.location)

        # set weather to sunny
        weather_preset, weather_preset_str = find_weather_presets()[0]
        weather_preset_str = weather_preset_str.replace(" ", "_")
        world.set_weather(weather_preset)
        simulation_identifier = (
            town_string + "_" + weather_preset_str + "_" + date_time_string
        )

        # create a vehicle
        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("vehicle.audi.tt")),
            start_pose,
        )
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # create camera and attach to vehicle
        cam_rgb_transform = carla.Transform(
            carla.Location(x=0.5, z=cg.height),
            carla.Rotation(pitch=-1 * cg.pitch_deg),
        )
        trafo_matrix_vehicle_to_cam = np.array(
            cam_rgb_transform.get_inverse_matrix()
        )
        bp = blueprint_library.find("sensor.camera.rgb")
        fov = cg.field_of_view_deg
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        camera_rgb = world.spawn_actor(
            bp, cam_rgb_transform, attach_to=vehicle
        )
        actor_list.append(camera_rgb)

        K = get_intrinsic_matrix(fov, width, height)
        min_jump, max_jump = 5, 10

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            frame = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Choose the next spawn_waypoint and update the car location.
                # ----- change lane with low probability
                if np.random.rand() > 0.9:
                    shifted = None
                    if spawn_waypoint.lane_change == carla.LaneChange.Left:
                        shifted = spawn_waypoint.get_left_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Right:
                        shifted = spawn_waypoint.get_right_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Both:
                        if np.random.rand() > 0.5:
                            shifted = spawn_waypoint.get_right_lane()
                        else:
                            shifted = spawn_waypoint.get_left_lane()
                    if shifted is not None:
                        spawn_waypoint = shifted
                # ----- jump forwards a random distance
                jump = np.random.uniform(min_jump, max_jump)
                next_waypoints = spawn_waypoint.next(jump)
                if not next_waypoints:
                    spawn_waypoint = get_random_spawn_point(m)
                else:
                    spawn_waypoint = random.choice(next_waypoints)

                # ----- randomly change yaw and lateral position
                spawn_transform = random_transform_disturbance(
                    spawn_waypoint.transform
                )
                vehicle.set_transform(spawn_transform)

                # Draw the display.
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                draw_image(display, image_rgb)
                display.blit(
                    font.render(
                        "% 5d FPS (real)" % clock.get_fps(),
                        True,
                        (255, 255, 255),
                    ),
                    (8, 10),
                )
                display.blit(
                    font.render(
                        "% 5d FPS (simulated)" % fps, True, (255, 255, 255)
                    ),
                    (8, 28),
                )

                # draw lane boundaries as augmented reality
                trafo_matrix_world_to_vehicle = np.array(
                    vehicle.get_transform().get_inverse_matrix()
                )
                trafo_matrix_global_to_camera = (
                    trafo_matrix_vehicle_to_cam @ trafo_matrix_world_to_vehicle
                )
                mat_swap_axes = np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
                )
                trafo_matrix_global_to_camera = (
                    mat_swap_axes @ trafo_matrix_global_to_camera
                )


                waypoint = m.get_waypoint(
                    vehicle.get_transform().location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                waypoints = [waypoint]
                waypoints = get_waypoints(waypoints, waypoint, side = "left")
                waypoints = get_waypoints(waypoints, waypoint, side = "right")
                waypoints = list(filter(lambda x: x is not None, waypoints))
                centers_list = []
                left_boundary_list = []
                right_boundary_list = []
                proj_center_list = []
                proj_left_bound_list = []
                proj_right_bound_list = []
                none_flags_list = [0]*len(waypoints) # 1 or 0
                for waypoint in waypoints:
                    center_list, left_boundary, right_boundary = create_lane_lines(
                        waypoint
                    )
                    centers_list.append(center_list)
                    left_boundary_list.append(left_boundary)
                    right_boundary_list.append(right_boundary)
                    if center_list is not None:
                        projected_center = project_polyline(
                        center_list, trafo_matrix_global_to_camera, K
                        ).astype(np.int32)
                        projected_left_boundary = project_polyline(
                            left_boundary, trafo_matrix_global_to_camera, K
                        ).astype(np.int32)
                        projected_right_boundary = project_polyline(
                            right_boundary, trafo_matrix_global_to_camera, K
                        ).astype(np.int32)   
                    else:
                        projected_center = projected_left_boundary = projected_right_boundary = None
                        none_flags_list[waypoints.index(waypoint)] = 1
                    proj_center_list.append(projected_center)
                    proj_left_bound_list.append(projected_left_boundary)
                    proj_right_bound_list.append(projected_right_boundary)
                
                if none_flags_list.count(1) == len(waypoints):
                    spawn_waypoint = get_random_spawn_point(m)
                    continue

                out_image_counter = 0
                for projected_center, projected_right_boundary, projected_left_boundary, idx in zip(proj_center_list, proj_right_bound_list, proj_left_bound_list, range(len(waypoints))):
                    if none_flags_list[idx] == 0:
                        right_check, r_noise_flag = check_inside_image(
                                projected_right_boundary, width, height
                            )
                        left_check, l_noise_flag = check_inside_image(
                                projected_left_boundary, width, height
                            )
                        if (not right_check) or (not left_check):
                            out_image_counter += 1
                        if r_noise_flag or l_noise_flag:
                            none_flags_list[idx] = 1
                            continue
                        else:
                            if len(projected_center) > 1:
                                pygame.draw.lines(
                                    display, (255, 136, 0), False, projected_center, 4
                                )
                            if len(projected_left_boundary) > 1:
                                pygame.draw.lines(
                                    display, (255, 0, 0), False, projected_left_boundary, 4
                                )
                            if len(projected_right_boundary) > 1:
                                pygame.draw.lines(
                                    display,
                                    (0, 255, 0),
                                    False,
                                    projected_right_boundary,
                                    4,
                                )
                if out_image_counter == len(waypoints):
                    spawn_waypoint = get_random_spawn_point(m)
                    continue

                in_lower_part_of_map = spawn_transform.location.y < 0

                if store_files:
                    filename_base = simulation_identifier + "_frame_{}".format(
                        frame
                    )
                    if in_lower_part_of_map:
                        if (
                            np.random.rand() > 0.1
                        ):  # do not need that many files from validation set
                            continue
                        filename_base += "_validation_set"
                    # image
                    image_out_path = os.path.join(
                        data_folder, filename_base + ".png"
                    )
                    save_img(image_rgb, image_out_path)
                    # label img
                    label_path = os.path.join(
                        data_folder, filename_base + "_label.png"
                    )
                    save_label_img(
                        proj_left_bound_list,
                        proj_right_bound_list,
                        none_flags_list,
                        label_path,
                    )
    
                for center_list in centers_list:
                    if center_list is not None:
                        curvature = get_curvature(center_list)
                        if curvature > 0.0005:
                            min_jump, max_jump = 1, 2
                        else:
                            min_jump, max_jump = 5, 10

                pygame.display.flip()
                frame += 1

    finally:

        print("destroying actors.")
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print("done.")


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
