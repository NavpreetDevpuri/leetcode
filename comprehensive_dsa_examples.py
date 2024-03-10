# Python Inbuilt Functionalities Examples

# List Examples
my_list = [1, 2, 3]
my_list.append(4)  # Append element
my_list.extend([5, 6])  # Extend list
print("List after append and extend:", my_list)

# Dictionary Examples
my_dict = {'a': 1, 'b': 2}
my_dict['c'] = 3  # Add new key-value pair
print("Dictionary items:", my_dict.items())

# Set Examples
my_set = {1, 2, 3}
my_set.add(4)  # Add element
print("Set after adding an element:", my_set)

# Tuple Examples
my_tuple = (1, 2, 3)
a, b, c = my_tuple  # Unpacking
print("Unpacked values:", a, b, c)

# String Examples
my_string = "Hello, World!"
reversed_string = my_string[::-1]  # Reverse string
print("Reversed String:", reversed_string)

# Deque Examples (from collections)
from collections import deque
my_deque = deque([1, 2, 3])
my_deque.appendleft(0)  # Append element to the left
print("Deque after appendleft:", list(my_deque))

# NamedTuple Example (from collections)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
pt = Point(1, -1)
print("NamedTuple Point:", pt.x, pt.y)

# Sorting Example
sorted_list = sorted([3, 1, 4, 1], reverse=True)
print("Sorted List:", sorted_list)

# Searching Example (Using bisect)
import bisect
sorted_nums = [1, 2, 4, 5]
position = bisect.bisect_left(sorted_nums, 3)
print("Position to insert 3 in sorted list:", position)

# Heap Queue Example (from heapq)
import heapq
heap = [5, 7, 9, 1, 3]
heapq.heapify(heap)
print("Min heap:", heap)
heapq.heappush(heap, 4)
print("Min heap after pushing 4:", heap)

# Permutations & Combinations Example (from itertools)
from itertools import permutations
print("Permutations of 'ABC':", list(permutations('ABC')))

# Lambda, Map & Filter Examples
squared = map(lambda x: x**2, [1, 2, 3])
print("Squared numbers using map:", list(squared))
filtered = filter(lambda x: x > 2, [1, 2, 3, 4])
print("Filtered numbers using filter:", list(filtered))

# Reduce Example (from functools)
from functools import reduce
summed = reduce(lambda x, y: x + y, [1, 2, 3, 4])
print("Summed using reduce:", summed)

# List Slicing Example
sliced_list = [0, 1, 2, 3, 4, 5][1:5:2]  # Start at index 1, end at index 5, step by 2
print("Sliced list:", sliced_list)

# Regular Expressions Example (from re)
import re
match = re.search(r'\d+', '123abc456')
print("First group of digits found using re:", match.group() if match else 'No match')

# Mathematics Example (from math)
import math
print("Square root of 16 using math:", math.sqrt(16))

# Memoization Example (using functools.lru_cache)
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print("Fibonacci of 10 using memoization:", fib(10))

# Continuing Python Inbuilt Functionalities Examples

# Comprehensions Example: List, Dictionary, and Set
list_comprehension = [x**2 for x in range(10) if x % 2 == 0]
print("List Comprehension:", list_comprehension)

dict_comprehension = {x: x**2 for x in range(5)}
print("Dictionary Comprehension:", dict_comprehension)

set_comprehension = {x for x in 'hello world' if x not in 'aeiou'}
print("Set Comprehension:", set_comprehension)

# Decorators Example
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

print("Decorator Example:")
say_hello()

# Regular Expressions Continued (from re)
import re
email_pattern = r"[\w\.-]+@[\w\.-]+"
email_example = "Please contact us at: support@example.com."
emails = re.findall(email_pattern, email_example)
print("Emails found using re:", emails)

# Math and Statistics (from math, statistics)
import statistics
data = [1.3, 2.7, 0.8, 4.1, 4.3, -0.1]
avg = statistics.mean(data)
print("Average of data using statistics:", avg)

# Using itertools for advanced iteration patterns
from itertools import product
product_example = list(product('AB', repeat=2))
print("Cartesian product using itertools:", product_example)

from itertools import cycle
cycling = cycle('AB')
cycling_list = [next(cycling) for _ in range(5)]
print("Cycle example using itertools:", cycling_list)

# Using functools for more advanced function manipulation
from functools import partial

def multiply(x, y):
    return x * y

# Creating a new function that multiplies by 2
dbl = partial(multiply, 2)
print("Using functools.partial:", dbl(4))

# Advanced String Manipulations
string_formatting = "Python {}".format("Rocks!")
print("String Formatting Example:", string_formatting)

formatted_string = f"1 + 2 is {1 + 2}"
print("Formatted String with f-strings:", formatted_string)

# Advanced Usage of Collections
from collections import Counter
counter_example = Counter('mississippi')
print("Counter Example:", counter_example)

from collections import ChainMap
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'b': 4}
chain_map_example = ChainMap(dict1, dict2)
print("ChainMap Example:", chain_map_example)

# Exploring bisect for more than just insertion points
import bisect
scores = [33, 99, 77, 70, 89, 90, 100]
grades = 'FDCBA'
breakpoints = [60, 70, 80, 90]
position = bisect.bisect(breakpoints, 85)
print("Using bisect for grading (85):", grades[position])

# Further Exploring Python Inbuilt Functionalities

# Using the 'any' and 'all' functions for iterables
test_any = [False, True, False]
test_all = [True, True, True]
print("Test any true:", any(test_any))
print("Test all true:", all(test_all))

# Working with 'enumerate' for looping with index
for index, value in enumerate(['a', 'b', 'c']):
    print(f"Index {index}: {value}")

# Advanced usage of 'zip' to iterate over two lists in parallel
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Using 'reversed' to iterate over a sequence in reverse
for i in reversed(range(1, 4)):
    print(f"Reversed range element: {i}")

# File handling basics: Writing and reading a file
# Writing to a file
with open("test.txt", "w") as f:
    f.write("Hello, Python!\n")

# Reading from a file
with open("test.txt", "r") as f:
    content = f.read()
print("File content:", content)

# Using 'filterfalse' from itertools to filter out elements returning false
from itertools import filterfalse
filtered_false = list(filterfalse(lambda x: x % 2 == 0, range(10)))
print("Filterfalse (odd numbers):", filtered_false)

# 'accumulate' from itertools to get accumulated sums
from itertools import accumulate
acc_sums = list(accumulate([1, 2, 3, 4, 5]))
print("Accumulate (sums):", acc_sums)

# Working with 'groupby' from itertools for grouping adjacent elements
from itertools import groupby
for key, group in groupby("AAAABBBCCDAABBB"):
    print(f"Key: {key}, Group: {''.join(group)}")

# 'chain' from itertools to iterate over multiple iterables as if they were a single sequence
from itertools import chain
combined = list(chain('ABC', 'DEF'))
print("Chain (combined):", combined)

# Advanced Dictionary Operations: Merging dictionaries
dict_a = {'a': 1, 'b': 2}
dict_b = {'b': 3, 'c': 4}
merged_dict = {**dict_a, **dict_b}
print("Merged Dictionary:", merged_dict)

# Using 'defaultdict' from collections for default values for missing keys
from collections import defaultdict
default_d = defaultdict(int)  # default integer value for missing keys
default_d['key1'] += 1
print("Defaultdict example:", dict(default_d))

# Advanced Set Operations: Set comprehensions and operations
even_nums = {x for x in range(10) if x % 2 == 0}
print("Even numbers (set comprehension):", even_nums)
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}
print("Union:", set_a | set_b)
print("Intersection:", set_a & set_b)
print("Difference:", set_a - set_b)

# Exploring 'math' module for constants and functions
print("Pi constant:", math.pi)
print("Cosine of pi/4:", math.cos(math.pi / 4))

# Using 'statistics' for basic statistical operations
data = [1, 2, 2, 3, 4, 4, 4, 5, 5, 6]
print("Median of data:", statistics.median(data))
print("Mode of data:", statistics.mode(data))

# Exploring Additional Python Inbuilt Functionalities

# Using 'os' and 'sys' modules for interacting with the operating system and the Python runtime environment
import os
import sys

print("Current working directory:", os.getcwd())
print("Python version:", sys.version)

# Working with 'datetime' for manipulating dates and times
from datetime import datetime, timedelta

now = datetime.now()
print("Current datetime:", now)
print("One week from now:", now + timedelta(weeks=1))

# 'fractions' and 'decimal' for precise arithmetic operations
from fractions import Fraction
from decimal import Decimal

fraction_sum = Fraction(1, 3) + Fraction(1, 3)
print("Fraction sum:", fraction_sum)
decimal_sum = Decimal('0.1') + Decimal('0.2')
print("Decimal sum (precise):", decimal_sum)

# 'random' module for generating random numbers and making random choices
import random

print("Random integer between 1 and 10:", random.randint(1, 10))
print("Random choice from a list:", random.choice(['apple', 'banana', 'cherry']))

# 'json' for parsing and generating JSON data
import json

json_data = '{"name": "John", "age": 30, "city": "New York"}'
parsed_json = json.loads(json_data)
print("Parsed JSON:", parsed_json)
generated_json = json.dumps(parsed_json, indent=4)
print("Generated JSON:", generated_json)

# 'hashlib' for hashing data (e.g., for secure storage or comparison of passwords)
import hashlib

hashed = hashlib.sha256("Python is awesome!".encode()).hexdigest()
print("SHA-256 Hashed:", hashed)

# Using 'argparse' for parsing command-line arguments in scripts
import argparse

# Uncomment to run argparse example - Note: This requires command-line execution
# parser = argparse.ArgumentParser(description="An argparse example")
# parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# args = parser.parse_args()
# print(args.accumulate(args.integers))

# 'subprocess' for spawning new processes, connecting to their input/output/error pipes, and obtaining their return codes
import subprocess

# Uncomment to run subprocess example - Note: This may require adjustments based on your operating system and installed programs
# result = subprocess.run(['echo', 'Hello from subprocess'], capture_output=True, text=True)
# print("Subprocess output:", result.stdout)

# 'threading' and 'multiprocessing' for executing code in parallel, allowing for concurrent execution
import threading

def print_numbers():
    for i in range(5):
        print(i)

# Uncomment to run threading example
# t = threading.Thread(target=print_numbers)
# t.start()
# t.join()

# This series of examples showcases the versatility of Python's standard library, from system interactions, date and time manipulations, numerical precision, random number generation, JSON parsing and generation, data hashing, command-line parsing, process spawning, to concurrent execution.
