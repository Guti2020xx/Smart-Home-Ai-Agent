import numpy as np
import pandas as pd

n_steps = 1000
data = []

#default home state
home_state = {
    "inside_temp" : 70,
    "outside_temp" : 80,
    "humidity" : 50,
    "occupancy" : 1
}


action_vectors = [
    "Activate dehumidifier", 
    "No action needed", 
    "Set home to eco mode", 
    "Turn on AC",
    "Turn on heater"
]

for i in range(n_steps):
    # Randomly vary outside temp and occupancy to simulate realistic conditions
    home_state["outside_temp"] += np.random.uniform(-0.5, 0.5)
    if np.random.rand() < 0.05:  # 5% chance occupancy changes
        home_state["occupancy"] = 1 - home_state["occupancy"]
    
    # Determine action based on current state
    if home_state["inside_temp"] < 68 and home_state["occupancy"]:
        action = "Turn on heater"
    elif home_state["inside_temp"] > 78 and home_state["occupancy"]:
        action = "Turn on AC"
    elif home_state["humidity"] > 70:
        action = "Activate dehumidifier"
    elif home_state["occupancy"] == 0:
        action = "Set home to eco mode"
    else:
        action = "No action needed"

    # Apply action to update home state
    if action == "Turn on heater":
        home_state["inside_temp"] += 2
    elif action == "Turn on AC":
        home_state["inside_temp"] -= 2
    elif action == "Activate dehumidifier":
        home_state["humidity"] -= 5
        if home_state["humidity"] < 0:
            home_state["humidity"] = 0

    # Save current state + action
    data.append({
        "inside_temp": home_state["inside_temp"],
        "outside_temp": home_state["outside_temp"],
        "humidity": home_state["humidity"],
        "occupancy": home_state["occupancy"],
        "action": action
    })

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("smart_home_data.csv", index=False)
print("Data generation complete! Shape:", df.shape)
