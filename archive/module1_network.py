"""
Module 1
Network Generation and Traffic Model

IEEE Paper:
Federated Learning Based AI-Driven Resource Scheduling
for 5G/6G Networks

Author: Prajwal Gupta

This module generates

• gNB deployment
• UE deployment
• CQI
• SINR
• Mobility
• Traffic demand

"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

# ----------------------------------------------------
# Simulation Parameters
# ----------------------------------------------------

np.random.seed(42)
random.seed(42)

NUM_GNBS = 20

MIN_USERS = 50
MAX_USERS = 200

CELL_RADIUS = 500          # metres
AREA_SIZE = 4000           # metres

BANDWIDTH = 100            # MHz
CARRIER_FREQ = 3.5e9

TIME_STEPS = 100

# ----------------------------------------------------
# Traffic Classes
# ----------------------------------------------------

TRAFFIC_TYPES = [
    "eMBB",
    "URLLC",
    "mMTC"
]

TRAFFIC_PROB = [
    0.55,
    0.25,
    0.20
]

# ----------------------------------------------------
# Dataclasses
# ----------------------------------------------------

@dataclass
class User:

    id: int

    x: float
    y: float

    speed: float

    traffic: str

    cqi: int

    sinr: float

    serving_gnb: int


@dataclass
class GNB:

    id: int

    x: float
    y: float

    users: list


# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------

def distance(x1,y1,x2,y2):

    return np.sqrt(
        (x1-x2)**2 +
        (y1-y2)**2
    )


# ----------------------------------------------------
# Deploy gNBs
# ----------------------------------------------------

def deploy_gnbs():

    gnbs=[]

    for i in range(NUM_GNBS):

        x=np.random.uniform(0,AREA_SIZE)

        y=np.random.uniform(0,AREA_SIZE)

        gnbs.append(
            GNB(
                id=i,
                x=x,
                y=y,
                users=[]
            )
        )

    return gnbs


# ----------------------------------------------------
# Random SINR
# ----------------------------------------------------

def generate_sinr():

    return np.random.uniform(-5,30)


# ----------------------------------------------------
# SINR -> CQI
# ----------------------------------------------------

def sinr_to_cqi(sinr):

    table=[

        -6.8,-4.8,-2.3,0,
        2.4,4.3,5.9,7.5,
        9.1,10.8,12.5,
        14.2,16,18,25,30

    ]

    for i,t in enumerate(table):

        if sinr<t:

            return max(1,i)

    return 15


# ----------------------------------------------------
# User Deployment
# ----------------------------------------------------

def deploy_users(gnbs):

    uid=0

    all_users=[]

    for gnb in gnbs:

        n=np.random.randint(
            MIN_USERS,
            MAX_USERS+1
        )

        for _ in range(n):

            angle=np.random.uniform(
                0,
                2*np.pi
            )

            r=np.random.uniform(
                0,
                CELL_RADIUS
            )

            x=gnb.x+r*np.cos(angle)

            y=gnb.y+r*np.sin(angle)

            speed=np.random.uniform(
                0,
                30
            )

            traffic=np.random.choice(
                TRAFFIC_TYPES,
                p=TRAFFIC_PROB
            )

            sinr=generate_sinr()

            cqi=sinr_to_cqi(sinr)

            user=User(

                id=uid,

                x=x,
                y=y,

                speed=speed,

                traffic=traffic,

                cqi=cqi,

                sinr=sinr,

                serving_gnb=gnb.id

            )

            uid+=1

            gnb.users.append(user)

            all_users.append(user)

    return all_users


# ----------------------------------------------------
# Mobility
# ----------------------------------------------------

def update_positions(users):

    for u in users:

        theta=np.random.uniform(
            0,
            2*np.pi
        )

        step=u.speed*0.5

        u.x+=step*np.cos(theta)

        u.y+=step*np.sin(theta)

        u.sinr=generate_sinr()

        u.cqi=sinr_to_cqi(u.sinr)


# ----------------------------------------------------
# Traffic Generation
# ----------------------------------------------------

def generate_traffic(users):

    traffic=[]

    for u in users:

        if u.traffic=="eMBB":

            demand=np.random.uniform(
                10,
                100
            )

        elif u.traffic=="URLLC":

            demand=np.random.uniform(
                1,
                10
            )

        else:

            demand=np.random.uniform(
                0.1,
                2
            )

        traffic.append(demand)

    return np.array(traffic)


# ----------------------------------------------------
# Visualisation
# ----------------------------------------------------

def plot_network(gnbs):

    plt.figure(figsize=(8,8))

    for g in gnbs:

        plt.scatter(
            g.x,
            g.y,
            marker="^",
            s=180,
            label="gNB" if g.id==0 else ""
        )

        xs=[u.x for u in g.users]

        ys=[u.y for u in g.users]

        plt.scatter(
            xs,
            ys,
            s=8,
            alpha=0.45,
            label="UE" if g.id==0 else ""
        )

    plt.grid(True)

    plt.xlabel("X Position (m)")

    plt.ylabel("Y Position (m)")

    plt.title("5G Network Deployment")

    plt.legend()

    plt.tight_layout()

    plt.show()


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__=="__main__":

    gnbs=deploy_gnbs()

    users=deploy_users(gnbs)

    print("="*50)

    print("Network Created")

    print("="*50)

    print("gNBs :",len(gnbs))

    print("Users:",len(users))

    embb=sum(
        u.traffic=="eMBB"
        for u in users
    )

    urllc=sum(
        u.traffic=="URLLC"
        for u in users
    )

    mmtc=sum(
        u.traffic=="mMTC"
        for u in users
    )

    print()

    print("Traffic Distribution")

    print("eMBB :",embb)

    print("URLLC:",urllc)

    print("mMTC :",mmtc)

    plot_network(gnbs)