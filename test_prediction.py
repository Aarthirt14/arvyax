import requests
import json

response = requests.post('http://localhost:8000/predict', json={
    'journal_text': 'Feeling stressed and overwhelmed by work',
    'stress_level': 5,
    'energy_level': 3,
    'sleep_hours': 6,
    'time_of_day': 'afternoon',
    'ambience_type': 'rain',
    'reflection_quality': 'clear'
})

print('Status:', response.status_code)
print('✓ SUCCESS!' if response.status_code == 200 else '✗ FAILED')
print()

if response.status_code == 200:
    d = response.json()
    print('PREDICTION RESULTS')
    print('=' * 50)
    print('State:', d.get('predicted_state'))
    print('Intensity:', d.get('predicted_intensity'), '/5')
    print('Confidence:', f"{d.get('confidence')*100:.0f}%")
    print('Action:', d.get('what_to_do'))
    print('Timing:', d.get('when_to_do'))
    print()
    print('Supportive Message:')
    print(d.get('supportive_message'))
else:
    print('Error:', response.json())
