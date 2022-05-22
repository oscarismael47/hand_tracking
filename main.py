import cv2
import mediapipe as mp
import numpy as np
from ursina import *


class Palm:
    def __init__(self,n_p=5):
        uvs = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        colors = (color.blue, color.blue, color.blue)
        self.palm_1 = [Entity(model=Mesh(vertices=[], triangles=[],uvs=uvs, normals=[], colors=colors,thickness=4),texture='white_cube') for _ in range(n_p)]
        self.palm_2 = [Entity(model=Mesh(vertices=[], triangles=[],uvs=uvs, normals=[], colors=colors,thickness=4),texture='white_cube') for _ in range(n_p)]
        self.n_p = n_p

    def update(self,verts_list):
        #[destroy(c) for palm_segment in self.palm_1 for c in  palm_segment.children]
        #[destroy(c) for palm_segment in self.palm_2 for c in  palm_segment.children]

        for v,verts in enumerate(verts_list):
            norms = ((0,0,-1),) * len(verts)    
            tris1 = (0,2,1)

            self.palm_1[v].model.vertices = verts
            self.palm_1[v].model.triangles = tris1
            self.palm_1[v].model.norms = norms
            self.palm_1[v].model.generate()

            tris2 = (0,1,2)
            self.palm_2[v].model.vertices = verts
            self.palm_2[v].model.triangles = tris2
            self.palm_2[v].model.norms = norms
            self.palm_2[v].model.generate()


class Hand:
    def __init__(self,type,color_point,height,width):
        self.joints =  [[0,1],[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],[2,5]]
        self.hand_points = [Entity(model='sphere', color=color_point, scale=0.5) for _  in range(21)]
        self.hand_joints = [Entity(model=Cylinder(resolution=5, radius=.2, start=0, height=1, direction=(0,0,1), mode='triangle'),scale=1, color=color.blue,texture="vertical_gradient") for _  in range(len(self.joints))]
        self.height,self.width= height,width
        self.palm_segs = [[0,13,17],
                      [0,9,13],
                      [0,5,9],
                      [0,1,5],
                      [1,2,5]]
        self.palm = Palm()

    def __draw_joint(self,point_1,point_2,joint):
        point_1_x,point_1_y,point_1_z = point_1.position 
        point_2_x,point_2_y,point_2_z = point_2.position
        x = point_2_x-point_1_x
        y = point_2_y-point_1_y
        z = point_2_z-point_1_z
        p = np.sqrt((x**2)+(y**2)+(z**2))
        joint.position = point_1.position
        joint.scale_z = p 
        joint.look_at(point_2, 'forward')  
    def update(self,points):
        hand_xyz = np.zeros(shape=(21,3),dtype=float)
        hand_xyz[:,0],hand_xyz[:,1],hand_xyz[:,2] = points[:,0]*self.width,points[:,1]*self.height,points[:,2]*100*2
        hand_xyz[:,0],hand_xyz[:,1]= self.width - hand_xyz[:,0], self.height - hand_xyz[:,1]
        for p,point in enumerate(hand_xyz):
            self.hand_points[p].position = (point[0],point[1],point[2])
        for j,joint in enumerate(self.joints):
            p1,p2 = joint
            self.__draw_joint(self.hand_points[p1],self.hand_points[p2],self.hand_joints[j])
        
        verts_list = []
        for p_seg in self.palm_segs:
            v1 = tuple(self.hand_points[p_seg[0]].position)
            v2 = tuple(self.hand_points[p_seg[1]].position)
            v3 = tuple(self.hand_points[p_seg[2]].position)
            verts = (v1, v2, v3)
            verts_list.append(verts)
        self.palm.update(verts_list)



def input(key,points):
    if key == 'escape':
        quit()


def update():
    for hand in hand_tracking.hands_data:
        hand_type = hand[0]
        hand_points = hand[1]

        if hand_type == 'Left':
            hand_left.update(hand_points)
        if hand_type == 'Right':
            hand_right.update(hand_points)



class Hand_tracking:
    def __init__(self):
    
        # For webcam input
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands_data = []

    def close_camera(self):
        self.cap.release()
    def process_frame(self,duration=1):
        with self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            success, image = self.cap.read()
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.qqq
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hands_data = []
                for hand_landmarks,hand_class in zip(results.multi_hand_landmarks,results.multi_handedness):
                    hand_class = hand_class.classification[0].label
                    keypoints = []
                    for data_point in hand_landmarks.landmark:
                        # X,Y,Z
                        keypoints.append([data_point.x,data_point.y,data_point.z])
                    keypoints = np.asarray(keypoints)    
                    hands_data.append([hand_class,keypoints])                         
                self.hands_data = hands_data
            #self._draw_hands()

                
    def _draw_hands(self,radius = 5,color = (255, 0, 0),thickness = -1):
        h,w,_ = self.image.shape
        for hand in self.hands_data:
            hand_type,hand = hand[0],hand[1]
            hand_xy = np.zeros_like(hand,dtype=int)
            hand_xy[:,0],hand_xy[:,1] = hand[:,0]*w,hand[:,1]*h
            for p,point in enumerate(hand_xy):
                self.image = cv2.circle(self.image, (point[0],point[1]), radius, color, thickness)
                self.image = cv2.putText(self.image, str(p), (point[0],point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        self.image = cv2.resize(self.image,(1300,1000))
        cv2.imshow('hand tracking', self.image)


height,width = 20,30
app = Ursina()
Entity(model='cube', color=color.white, scale=1)
camera.position = (width//2,height//2,-100)
window.borderless = False


hand_left = Hand(type='Left',color_point=color.green,height=height,width=width)
hand_right = Hand(type='Right',color_point=color.red,height=height,width=width)

hand_tracking = Hand_tracking()
hand_tracking_sequence = Sequence(0.01,Func(hand_tracking.process_frame,duration=0.01),loop=True)
hand_tracking_sequence.start()


app.run()