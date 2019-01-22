from math import pi, atan, atan2, asin
def quaternionToEulerAngles(x, y, z, w) :
    # X axis rotation
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    # Y axis rotation
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1 :
    	pitch = pi/2 * (sinp/sinp) # use 90 degrees if out of range
    else :
        pitch = asin(sinp)

    # Z axis rotation
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
