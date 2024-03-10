import bisect
scores = [33, 99, 77, 70, 89, 90, 100]
grades = 'FDCBA'
breakpoints = [60, 70, 80, 90]
position = bisect.bisect(breakpoints, 90)
print("Using bisect for grading (85):", grades[position])
