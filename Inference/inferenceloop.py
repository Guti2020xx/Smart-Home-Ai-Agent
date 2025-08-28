import onnxruntime as ort 
import numpy as np
import time
import random

# Load ONNX model
ort_session = ort.InferenceSession(r"../Model/smart_home_transformer.onnx")

# Define action vectors and features
action_vectors = ["Activate dehumidifier", "No action needed", "Set home to eco mode", "Turn on AC", "Turn on heater"]
features = ["inside_temp", "outside_temp", "humidity", "occupancy"]

def explain_action(action, home_state):
    inside = home_state["inside_temp"]
    outside = home_state["outside_temp"]
    humidity = home_state["humidity"]
    occupancy = home_state["occupancy"]

    explanation = f"Inside temp: {inside:.1f}°F, Outside temp: {outside:.1f}°F, "
    explanation += f"Humidity: {humidity:.1f}%, "
    explanation += f"{'Someone is home' if occupancy else 'No one is home'}. "

    if action == "Turn on heater":
        if occupancy:
            explanation += ("Since the inside temperature is below comfort level and someone is present, "
                            "the system decided to turn on the heater to warm up the room.")
        else:
            explanation += ("The inside temperature is below comfort level, but no one is home. "
                            "The system still decided to turn on the heater, perhaps for pre-heating or safety.")
    elif action == "Turn on AC":
        if occupancy:
            explanation += ("The inside temperature is above the preferred comfort range and someone is home, "
                            "so the system activated the AC to cool down the room.")
        else:
            explanation += ("The inside temperature is above the preferred comfort range, but no one is home. "
                            "The system activated the AC, possibly to prevent overheating or for pre-cooling.")
    elif action == "Activate dehumidifier":
        explanation += ("The humidity inside is high, which can be uncomfortable or unhealthy, "
                        "so the dehumidifier is turned on to reduce moisture.")
    elif action == "Set home to eco mode":
        if occupancy == 0:
            explanation += ("No one is currently home, so the system is conserving energy by setting "
                            "the home to eco mode instead of heating or cooling.")
        else:
            explanation += ("Someone is home, but the system is prioritizing energy savings by setting "
                            "the home to eco mode instead of heating or cooling.")
    else:
        explanation += ("All conditions are within acceptable comfort ranges, "
                        "so no action is needed at this time.")

    return explanation

# Initialize home state
home_state = {
    "inside_temp": 70,
    "outside_temp": 85,
    "humidity": 50,
    "occupancy": 1
}

# Pre-fill the sequence with the *real* initial state
sequence = []
for _ in range(14):
    sequence.append([home_state[f] for f in features])

# Get ONNX input/output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

print("Smart Home Automation System - ONNX Inference\n")
print(f"Initial state: {home_state}\n")

for step in range(10):
    # Randomly flip occupancy 15% of the time
    if random.random() < 0.4:
        home_state["occupancy"] = 1 - home_state["occupancy"]

    # Add current state to sequence (keeps actual occupancy)
    sequence.append([home_state[f] for f in features])
    sequence = sequence[-15:]  # keep only last 15

    # Prepare input
    input_seq = np.array([sequence], dtype=np.float32)

    # Run ONNX inference
    logits = ort_session.run([output_name], {input_name: input_seq})[0]
    action_idx = np.argmax(logits)
    action = action_vectors[action_idx]

    # Print results
    print(f"Step {step+1} - Action: {action}")
    print(f"Explanation: {explain_action(action, home_state)}\n")

    # Update home state based on action
    if action == "Turn on heater":
        home_state["inside_temp"] += 2
    elif action == "Turn on AC":
        home_state["inside_temp"] -= 2
    elif action == "Activate dehumidifier":
        home_state["humidity"] -= 5
        home_state["humidity"] = max(0, home_state["humidity"])
    elif action == "Set home to eco mode":
        temp_diff = home_state["outside_temp"] - home_state["inside_temp"]
        home_state["inside_temp"] += temp_diff * 0.05  # drift toward outside

    # Optional: natural humidity rise every 3 steps
    if step % 3 == 0:
        home_state["humidity"] += 2
        home_state["humidity"] = min(100, home_state["humidity"])

    # Show updated state
    print(f"Updated state: Inside temp: {home_state['inside_temp']:.1f}°F, "
          f"Humidity: {home_state['humidity']:.1f}%, "
          f"Occupancy: {'Yes' if home_state['occupancy'] else 'No'}")
    print("-" * 80)

    time.sleep(0.5)

print("\nSimulation complete!")
print(f"Final state: {home_state}")
