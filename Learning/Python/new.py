# import datetime as dt

# today = dt.date.today();
# anotherday = dt.date(2016, 2, 29)
# print("Hello World today is ", today.day, " ", today.month)
# print("Hello World that day was ", anotherday.day, " ", anotherday.month)

# def date_time(a=6):
#     """ THis is a docstring"""
#     if(a)< 5:
#         return True
        


# print(date_time.__doc__); 

import matplotlib.pyplot as plt

x1 = [3, 5, 7]
y1 = [18, 2, 8]

# plt.figure()
# plt.plot(x,y)

# plt.title("Basic Line Graph")
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")

# plt.show()

# x = [6, 2, 9]
# y = [10, 3, 5]

# # Configure grid: 1 Row, 2 Distinct Columns -> Select Index position 1
# plt.subplot(3, 1, 1)
# plt.plot(x1, y1)
# plt.title("First Subplot")

# # Select Index position 2 within the same horizontal row array
# plt.subplot(3, 1, 3)
# plt.bar(y, x) # Interchanging values
# plt.title("Second Subplot")
# plt.show()

import numpy as np

# t = np.arange(0.0, 2.0, 0.01)
# s = np.sin(2 * np.pi * t)
# c = np.cos(2 * np.pi * t)

# fig , ax = plt.subplots()

# ax.plot(t, s, color='red', linestyle='-', label='Sine Wave')
# ax.plot(t, c, color='blue', linestyle='--', label='Cosine Wave')

# ax.set_title("Trigonometric Waveforms")
# ax.legend(loc="upper right")
# ax.grid(True)

# plt.show()

from mpl_toolkits.mplot3d import Axes3D

# # Initializing multi-dimensional positional coordinates
# x_coords = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y_coords = np.array([3, 4, 6, 11, 2, 5, 8, 9, 1, 7])
# z_coords = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# fig = plt.figure()
# # Instantiating structural 3D projection spaces
# # ax = fig.add_subplot(111, projection='3d')

# # ax.plot(x_coords, y_coords, z_coords)
# # plt.show()

# # 2D Data Scatter mapping syntax
# plt.scatter(x_coords, y_coords, color='red', marker='o')
# m, c = np.polyfit(x_coords, y_coords, 1)

# # 4. Plot the trendline using the calculated equation
# plt.plot(x_coords, m*x_coords + c, color='red', linewidth=2, label='Regression Line')
# plt.title("Data Scatter Distribution")
# plt.show()


import pandas as pd

# my_list = [1,5,2,5,8,3]
# int_series = pd.Series(my_list, index = [101,201,301,401,501,601])
# print(int_series[301])

# # Creating a fixed-frequency Datetime Index
# date_series = pd.date_range(start="01/01/2022", end="01/04/2022")

# # Pulling out underlying datetime attributes
# print(date_series.year)             # Outputs the structural year values
# print(date_series.month_name())     # Evaluates text-based names (e.g., 'January')
# print(date_series.day_of_year)      # Identifies absolute day counts within that calendar year
# student_dict = {
#     "student_name": ["Alex", "Anna", "John", "Tim", "Sid"],
#     "math_score": [10, 9, 9, 8, 10]
# }
# student_df = pd.DataFrame(student_dict, index=[1, 2, 3, 4, 5])
# print(student_df)

print = [1,2,3,4]