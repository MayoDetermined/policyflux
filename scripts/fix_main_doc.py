from pathlib import Path

path = Path('main.py')
text = path.read_text(encoding='utf-8')
start_marker = "        - influence_matrix: numpy array of shape (num_actors, num_actors) \n"
end_marker = "        - ipm_params: Dictionary with 'salience' and 'threshold' arrays for IPM voting\n"
start = text.find(start_marker)
end = text.find(end_marker)
if start == -1 or end == -1:
    raise SystemExit('Markers not found in main.py; manual cleanup needed')
replacement = (
    "        - influence_matrix: numpy array of shape (num_actors, num_actors) \n"
    "        - ideal_point_model: Trained IdealPointModel instance (or None if PyTorch unavailable)\n"
    "        - ipm_params: Dictionary with 'salience' and 'threshold' arrays for IPM voting\n"
)
text = text[:start] + replacement + text[end:]
path.write_text(text, encoding='utf-8')
