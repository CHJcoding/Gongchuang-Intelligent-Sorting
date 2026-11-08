import time, numpy as np, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

class N(Node):
    def __init__(self):
        super().__init__("cmp_sharp")
        self.raw = None
        self.ann = None
        self.raw_n = 0
        self.ann_n = 0
        self.pairs = []
        self.create_subscription(CompressedImage, "/image", self.cb_raw, QOS)
        self.create_subscription(CompressedImage, "/image_annotated", self.cb_ann, QOS)
        self.create_timer(0.05, self.tick)

    def cb_raw(self, m):
        self.raw = np.frombuffer(bytes(m.data), np.uint8)
        self.raw_n += 1
        self._try()

    def cb_ann(self, m):
        self.ann = np.frombuffer(bytes(m.data), np.uint8)
        self.ann_n += 1
        self._try()

    def _try(self):
        if self.raw is None or self.ann is None:
            return
        a, b = self.raw, self.ann
        self.raw = None
        self.ann = None
        ia = cv2.imdecode(a, cv2.IMREAD_COLOR)
        ib = cv2.imdecode(b, cv2.IMREAD_COLOR)
        if ia is None or ib is None:
            return
        ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
        self.pairs.append((len(a), len(b), ia.shape, ib.shape,
                           float(cv2.Laplacian(ga, cv2.CV_64F).var()),
                           float(cv2.Laplacian(gb, cv2.CV_64F).var())))
        print("pair %d: raw %d B lap=%.1f | ann %d B lap=%.1f  raw%s ann%s" % (
            len(self.pairs), len(a), self.pairs[-1][4], len(b), self.pairs[-1][5],
            ia.shape, ib.shape), flush=True)

    def tick(self):
        if len(self.pairs) >= 20:
            raise KeyboardInterrupt

def main():
    rclpy.init()
    n = N()
    print("waiting on /image and /image_annotated ...", flush=True)
    t0 = time.time()
    try:
        while len(n.pairs) < 20 and time.time() - t0 < 25:
            rclpy.spin_once(n, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    p = n.pairs
    print("---")
    print("raw_msgs=%d ann_msgs=%d pairs=%d" % (n.raw_n, n.ann_n, len(p)))
    if p:
        la = sorted(x[4] for x in p)
        lb = sorted(x[5] for x in p)
        med = lambda v: v[len(v)//2]
        print("median lap: /image=%.1f  /image_annotated=%.1f  ratio=%.3f" % (med(la), med(lb), med(lb)/max(med(la),1e-9)))
        print("bytes: /image=%d  ann=%d" % (med(sorted(x[0] for x in p)), med(sorted(x[1] for x in p))))
    n.destroy_node()
    rclpy.shutdown()

main()
