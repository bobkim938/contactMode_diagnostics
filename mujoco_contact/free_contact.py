import mujoco
import mujoco.viewer
import time

import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("mujoco_contact/models/wall_contact.xml")
data = mujoco.MjData(model)

log = open("mujoco_contact/log_freeContact.txt", "w")

# Initialize lists to store the data
times = []
qpos_list = []
qvel_list = []
ncon_list = []
normal_force = []
tangential_force = []


with mujoco.viewer.launch_passive(model, data) as viewer:
    # x-axis towards the wall, y-axis along the wall
    for i in range(1000):
        data.ctrl[0] = 10.0
        # data.ctrl[1] = 50.0

        mujoco.mj_step(model, data)
        viewer.sync()

        if data.ncon > 0:
            Fn_total = 0.0
            Ft_total = 0.0
            for contact in range(data.ncon): # contact force extraction
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact, contact_force)
                Fn_total += abs(contact_force[0])
                Ft_total += np.linalg.norm(contact_force[1:3])

            normal_force.append(Fn_total)
            tangential_force.append(Ft_total)

        else:
            normal_force.append(0.0)
            tangential_force.append(0.0)

        log.write(
            f"time={data.time:.3f}, "
            f"qpos={data.qpos.copy()}, "
            f"qvel={data.qvel.copy()}, "
            f"ncon={data.ncon}, "
            f"normal_force={normal_force[-1]:.3f}, "
            f"tangential_force={tangential_force[-1]:.3f}\n"
        )
        log.flush()

        times.append(data.time)
        qpos_list.append(data.qpos.copy())
        qvel_list.append(data.qvel.copy())
        ncon_list.append(data.ncon)

        log.flush()
        time.sleep(0.002)

fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

axs[0].plot(times, qpos_list)
axs[0].set_ylabel("qpos")

axs[1].plot(times, qvel_list)
axs[1].set_ylabel("qvel")

axs[2].plot(times, normal_force)
axs[2].set_ylabel("normal_force")

axs[3].plot(times, tangential_force)
axs[3].set_ylabel("tangential_force")
axs[3].set_xlabel("time")

axs[4].plot(times, ncon_list)
axs[4].set_ylabel("ncon")
axs[4].set_xlabel("time")



plt.tight_layout()
plt.savefig("mujoco_contact/freeContact.png")



