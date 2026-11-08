#!/usr/bin/env python3
import numpy as np, cv2, time, sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
frames=[]
def cb(m):
    frames.append(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR))
rclpy.init(); n=Node('cap'); n.create_subscription(CompressedImage,'/image',cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
want = int(sys.argv[1]) if len(sys.argv)>1 else 6
t0=time.time(); saved=0
while saved < want and time.time()-t0 < 40:
    rclpy.spin_once(n, timeout_sec=0.2)
    if len(frames) > saved:
        cv2.imwrite('/tmp/green_%02d.jpg' % saved, frames[-1]); saved+=1
        time.sleep(0.7)
print('saved', saved, 'frames', len(frames))
rclpy.shutdown()