#!/usr/bin/env python3
import numpy as np, cv2, rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
frames = []
def cb(m):
    frames.append(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR))
rclpy.init(); node = Node('diag'); node.create_subscription(CompressedImage, '/image', cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
t0 = time.time()
while time.time() - t0 < 15 and not frames:
    rclpy.spin_once(node, timeout_sec=0.2)
bgr = frames[-1]
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = hsv[:,:,0].astype(np.int32), hsv[:,:,1].astype(np.int32), hsv[:,:,2].astype(np.int32)
print('dtype', hsv.dtype, 'shape', hsv.shape, 'frames', len(frames))
print('mean BGR', bgr.reshape(-1,3).mean(0).round(1), 'mean HSV', H.mean().round(1), S.mean().round(1), V.mean().round(1))
print('S p50/p90/p99', np.percentile(S,50), np.percentile(S,90), np.percentile(S,99))
print('V p50/p90/p99', np.percentile(V,50), np.percentile(V,90), np.percentile(V,99))
print('cnt S54-182 & V114-255 =', int(((S>=54)&(S<=182)&(V>=114)).sum()))
print('cnt S>=70 & V>=90 =', int(((S>=70)&(V>=90)).sum()))
print('cnt H>=160 =', int((H>=160).sum()), ' H<=14 =', int((H<=14).sum()))
m = ((S>=54)&(V>=114)).astype(np.uint8)*255
ncc, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
order = sorted(range(1, ncc), key=lambda i: -stats[i,4])[:3]
for i in order:
    sel = lab==i
    print('comp area=%d bbox=%s H=%d S=%d V=%d' % (stats[i,4], tuple(stats[i][:4]),
        np.median(H[sel]), np.median(S[sel]), np.median(V[sel])))
cv2.imwrite('/tmp/diag.jpg', bgr)
rclpy.shutdown()
