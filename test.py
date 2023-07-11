
from trace_util.convert_trace import csv_to_dict, get_delta_progress


csv_to_dict()
result = csv_to_dict()
print(get_delta_progress(0.8, *result["mnist"][16, 8]))
