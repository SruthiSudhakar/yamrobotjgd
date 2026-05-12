"""List connected Intel RealSense devices (name + serial number).

Use the printed serials with `keyboard_teleop_planar.py --rs-serial-cam0 ... --rs-serial-cam1 ...`
to pin which physical camera is cam0 vs cam1 across USB re-plugs.
"""

import pyrealsense2 as rs


def main() -> None:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [{i}] {name}  serial={serial}  fw={fw}")


if __name__ == "__main__":
    main()
