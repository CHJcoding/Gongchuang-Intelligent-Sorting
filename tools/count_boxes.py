import time, rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets

def rois(t):
    r = getattr(t, 'rois', None)
    if r: return list(r)
    out=[]
    for tr in (getattr(t,'tracks',None) or []): out += list(getattr(tr,'rois',[]) or [])
    return out

class C(Node):
    def __init__(self):
        super().__init__('cnt')
        self.counts=[]
        self.create_subscription(PerceptionTargets,'/hobot_dnn_detection',self.cb,10)
    def cb(self,m):
        n=0
        for t in m.targets: n+=len(rois(t))
        self.counts.append(n)

rclpy.init(); n=C(); dl=time.time()+45
while time.time()<dl and len(n.counts)<60: rclpy.spin_once(n,timeout_sec=0.1)
c=n.counts
import statistics as st
print('messages=%d' % len(c))
print('boxes/message: min=%d max=%d mean=%.1f median=%.1f' % (min(c),max(c),st.mean(c),st.median(c)))
print('empty_msgs=%d' % sum(1 for x in c if x==0))
print('first20=%s' % c[:20])
n.destroy_node(); rclpy.shutdown()
