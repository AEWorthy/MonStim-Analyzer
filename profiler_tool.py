# To profile, use command 'python -m cProfile -o profiler_output.txt main.py'

import pstats

with open("profiler_output_readable.txt", "w") as f:
    stats = pstats.Stats("profiler_output.txt", stream=f)
    stats.sort_stats("cumtime")  # You can also sort by 'tottime' or 'calls'
    stats.print_stats()


# # Use below code to profile a function
# import cProfile
# import pstats
# import csv

# # Profile your code
# profiler = cProfile.Profile()
# profiler.enable()
# # Call the function you want to profile
# your_function()
# profiler.disable()

# # Write the results to a CSV file
# with open("profile_output.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Function", "Calls", "Total Time", "Per Call", "Cumulative Time", "Per Call Cumulative"])
    
#     stats = pstats.Stats(profiler)
#     for func, stat in stats.stats.items():
#         name = pstats.func_std_string(func)
#         cc, nc, tt, ct, callers = stat
#         writer.writerow([name, cc, tt, tt / cc if cc else 0, ct, ct / cc if cc else 0])