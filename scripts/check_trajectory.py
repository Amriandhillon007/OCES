# save as check_trajectory.py
import pickle

try:
    with open("phase11_trajectories.pkl", "rb") as f:
        data = pickle.load(f)
    print("✅ File is valid!")
    print(f"   Type: {type(data)}")
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        
        # Check what's inside
        if 'saved_states' in data:
            print(f"   Number of saved states: {len(data['saved_states'])}")
        if 'history' in data:
            print(f"   History keys: {list(data['history'].keys())}")
            for key in data['history']:
                if data['history'][key]:
                    print(f"      {key}: last value = {data['history'][key][-1]:.3f}")
                    
except Exception as e:
    print(f"❌ Error: {e}")
    