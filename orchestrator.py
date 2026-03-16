import os
import time
from openai import OpenAI

# Initialize the OpenAI client pointing to the Nebius Token Factory
# Replace with your actual Nebius Token Factory API Key once available
api_key = os.environ.get("NEBIUS_API_KEY", "YOUR_API_KEY_HERE")

client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=api_key,
)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# System prompt defines the Orchestrator's role and the strictly defined workflow
SYSTEM_PROMPT = """
You are the Orchestrator Agent for a Unitree G1 robotic arm automating a lab workflow.
The required workflow sequence is STRICTLY:
1. Extract (Soil sample extraction)
2. Dilute (Precise liquid dilution)
3. Plate (Slide plating)
4. Transfer (Microscope transfer)

Based on the current status update provided, determine the NEXT logical step to execute.
Respond ONLY with the exact single word name of the next step (Extract, Dilute, Plate, Transfer), or "Done" if the workflow is fully complete (Microscope transfer has been completed).
Do not provide additional explanation or conversation.
"""

def determine_next_step(status_update: str) -> str:
    """
    Calls the Nebius Token Factory LLM pipeline to reason about the current state
    and return the next required workflow step.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current Status: {status_update}\nWhat is the next step?"}
            ],
            temperature=0.0, # Low temperature for highly deterministic output
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with Nebius Token Factory: {e}")
        return "Error"

def main():
    print("==================================================")
    print("Orchestrator Agent Online (Powered by Nebius AI)")
    print("LLM Backend:", MODEL_NAME)
    print("==================================================\n")
    
    # Simulated status updates throughout the lab workflow
    simulated_statuses = [
        "System initialized. G1 arm is calibrated and at the home position above the lab bench. No tasks have been started.",
        "Soil sample successfully extracted from the vial and deposited into the mixing beaker.",
        "Liquid diluent successfully added and thoroughly mixed with the soil sample.",
        "Sample droplet successfully plated cleanly onto the center of the glass slide.",
        "Glass slide securely transferred to the microscope stage and released into clamps."
    ]
    
    for i, status in enumerate(simulated_statuses):
        print(f"--- Cycle {i+1} ---")
        print(f"[Sensors/State] '{status}'")
        
        print("[Orchestrator] Reasoning engine determining next step...")
        
        if api_key == "YOUR_API_KEY_HERE":
            # Bypass API call to avoid failure during logic review
            print("[WARNING] API Key missing. Skipping live Nebius API call.")
            # Hardcoded simulation of LLM responses matching the logic for demonstration
            expected_responses = ["Extract", "Dilute", "Plate", "Transfer", "Done"]
            next_step = expected_responses[i]
        else:
            # Call Nebius LLM
            next_step = determine_next_step(status)
        
        print(f"[Orchestrator] Decided Next Step: >> {next_step} <<")
        
        if next_step == "Done":
            print("\n[Orchestrator] Workflow sequence logic complete. Stopping system.")
            break
            
        print(f"[Orchestrator] Delegating [{next_step}] task to Kinematics & Vision Sub-Agents...\n")
        time.sleep(1.5) # Simulate execution time
        
    print("\nOrchestrator shutdown.")

if __name__ == "__main__":
    main()
